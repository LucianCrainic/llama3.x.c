#include "loader.h"
#include "workers.h"
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum { COPY_CHUNK_BYTES = 64 << 20, CACHE_LINE = 64 };

static const char *active_mode = "file";

const char *placement_mode(void) {
  const char *mode = getenv("LLAMA_PLACEMENT");
  if (!mode || !*mode)
    return "file";
  if (!strcmp(mode, "file") || !strcmp(mode, "ddr") || !strcmp(mode, "hbm"))
    return mode[0] == 'f' ? "file" : mode[0] == 'd' ? "ddr" : "hbm";
  fail("LLAMA_PLACEMENT must be file, ddr or hbm");
}

const char *placement_active(void) {
  return active_mode;
}

size_t placement_hbm_budget(void) {
  const char *value = getenv("LLAMA_HBM_BUDGET_GIB");
  if (!value || !*value)
    return (size_t)56 << 30;
  char *end;
  double gib = strtod(value, &end);
  if (*end || !(gib > 0) || gib > 1 << 20)
    fail("LLAMA_HBM_BUDGET_GIB must be a positive number of GiB");
  return (size_t)(gib * (double)(1 << 30));
}

/* ---- layout ----------------------------------------------------------------------------------------------------- */

int weights_plan(const TensorManifest *m, WeightSegment *seg, size_t *weights_bytes, size_t *embedding_bytes) {
  const TensorDesc *cls = manifest_find(m, T_WCLS, -1);
  int tied = cls && cls->alias_of >= 0;
  int count = 0;
  size_t cursor[2] = {0, 0}, end = 0;
  for (int i = 0; i < m->count; i++) {
    const TensorDesc *t = &m->t[i];
    if (t->alias_of >= 0)
      continue;
    int embedding = t->role == T_EMBED && !tied;
    if (count && !seg[count - 1].embedding == !embedding && t->offset == end) {
      seg[count - 1].bytes += t->bytes;
    } else {
      if (count == WEIGHT_SEGMENTS)
        fail("tensor manifest needs more than %d placement segments", WEIGHT_SEGMENTS);
      /* Keep the offset modulo the cache line, so the owned copy is aligned as the file mapping is. */
      size_t pad = (t->offset % CACHE_LINE + CACHE_LINE - cursor[embedding] % CACHE_LINE) % CACHE_LINE;
      seg[count] = (WeightSegment){.src_begin = t->offset, .bytes = t->bytes, .embedding = embedding,
                                   .dst_offset = checked_add(cursor[embedding], pad)};
      count++;
    }
    end = t->offset + t->bytes;
    /* The cursor of a class is the end of its last segment, grown by every tensor appended to it. */
    cursor[embedding] = seg[count - 1].dst_offset + seg[count - 1].bytes;
  }
  *weights_bytes = cursor[0];
  *embedding_bytes = cursor[1];
  return count;
}

const char *weights_at(const WeightStore *ws, size_t offset) {
  if (!ws->nseg)
    return ws->mapped + offset;
  for (int i = 0; i < ws->nseg; i++) {
    const WeightSegment *s = &ws->seg[i];
    if (offset >= s->src_begin && offset - s->src_begin < s->bytes)
      return (const char *)ws->region[s->embedding].data + s->dst_offset + (offset - s->src_begin);
  }
  fail("file offset %zu is not covered by any placement segment", offset);
}

/* ---- copy ------------------------------------------------------------------------------------------------------- */

typedef struct {
  int fd;
  const WeightStore *ws;
  size_t total, chunk, max_read;
  int *status; /* per worker: 0, an errno, or -1 for an unexpected end of file */
} CopyJob;

/* Reads exactly `length` bytes at `offset`, retrying short reads and EINTR. */
static int read_exact(int fd, char *dst, size_t length, size_t offset, size_t max_read) {
  while (length) {
    size_t want = length < max_read ? length : max_read;
    ssize_t got = pread(fd, dst, want, (off_t)offset);
    if (got < 0) {
      if (errno == EINTR)
        continue;
      return errno;
    }
    if (got == 0)
      return -1;
    dst += got;
    offset += (size_t)got;
    length -= (size_t)got;
  }
  return 0;
}

/* Each worker owns one contiguous slice of the bytes of all segments laid end to end. */
static void copy_slice(const void *job_, int worker, int workers) {
  const CopyJob *job = job_;
  size_t lo = (size_t)((unsigned long long)job->total * (unsigned)worker / (unsigned)workers);
  size_t hi = (size_t)((unsigned long long)job->total * (unsigned)(worker + 1) / (unsigned)workers);
  size_t base = 0;
  for (int i = 0; i < job->ws->nseg && base < hi && !job->status[worker]; i++) {
    const WeightSegment *s = &job->ws->seg[i];
    size_t end = base + s->bytes;
    if (end > lo) {
      size_t from = lo > base ? lo - base : 0, to = hi < end ? hi - base : s->bytes;
      char *dst = (char *)job->ws->region[s->embedding].data + s->dst_offset;
      for (size_t at = from; at < to && !job->status[worker];) {
        size_t piece = to - at < job->chunk ? to - at : job->chunk;
        job->status[worker] = read_exact(job->fd, dst + at, piece, s->src_begin + at, job->max_read);
        at += piece;
      }
    }
    base = end;
  }
}

static size_t env_size(const char *name, size_t fallback) {
  const char *value = getenv(name);
  if (!value || !*value)
    return fallback;
  char *end;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (*end || !parsed)
    fail("%s must be a positive integer", name);
  return (size_t)parsed;
}

/* ---- placement -------------------------------------------------------------------------------------------------- */

typedef struct {
  int weights_node, embedding_node; /* -1: unbound portable memory */
} Nodes;

static Nodes choose_nodes(const char *mode, int need_embedding_region) {
  Nodes nodes = {-1, -1};
  const Topology *topology;
  char err[512];
  int hbm = !strcmp(mode, "hbm");
  if (numa_topology_try(&topology, err, sizeof(err)) || numa_check_placement(topology, 1, mode, err, sizeof(err)) ||
      (hbm && need_embedding_region && numa_check_placement(topology, 1, "ddr", err, sizeof(err)))) {
    if (placement_strict(mode))
      fail("LLAMA_PLACEMENT=%s cannot be honoured: %s (LLAMA_PLACEMENT_STRICT=0 falls back to unbound memory)", mode,
           err);
    LOG(LOG_WARN, "LLAMA_PLACEMENT=%s: %s; using unbound memory", mode, err);
    return nodes;
  }
  nodes.weights_node = hbm ? topology->socket[0].hbm_node : topology->socket[0].ddr_node;
  nodes.embedding_node = topology->socket[0].ddr_node;
  return nodes;
}

static void reserve(NumaRegion *r, size_t bytes, int node, RegionRole role, const char *mode) {
  char err[512];
  if (region_reserve_try(r, bytes, 0, node, role, region_thp_default(), err, sizeof(err)) == 0)
    return;
  if (node < 0 || placement_strict(mode))
    fail("cannot reserve %zu bytes for the %s region: %s", bytes, region_role_name(role), err);
  LOG(LOG_WARN, "%s; using unbound memory", err);
  region_reserve(r, bytes, 0, -1, role, 0);
}

static void verify_region(NumaRegion *r, const char *mode) {
  if (r->node < 0)
    return;
  PageCheck check = {0};
  if (region_verify(r, region_verify_samples_default(), &check) != 0) {
    if (placement_strict(mode))
      fail("page placement of the %s region on node %d cannot be verified on this system", region_role_name(r->role),
           r->node);
    LOG(LOG_WARN, "page placement of the %s region cannot be verified", region_role_name(r->role));
    return;
  }
  size_t thp = 0;
  if (region_thp_bytes(r, &thp) != 0)
    thp = 0;
  region_record_verification(r, &check, thp);
  LOG(LOG_DEBUG, "%s region on node %d: %zu pages queried, %zu on node, %zu elsewhere, %zu missing",
      region_role_name(r->role), r->node, check.queried, check.on_node, check.wrong_node, check.missing);
  if ((check.wrong_node || check.missing || check.errors) && placement_strict(mode))
    fail("%s region: %zu of %zu queried pages are not on node %d (%zu elsewhere, %zu missing, %zu errors)",
         region_role_name(r->role), check.queried - check.on_node, check.queried, r->node, check.wrong_node,
         check.missing, check.errors);
}

void weights_open(WeightStore *ws, int fd, float **mapped, size_t file_bytes, const TensorManifest *m) {
  const char *mode = placement_mode();
  *ws = (WeightStore){.mode = mode, .mapped = (const char *)*mapped};
  active_mode = mode;
  if (!strcmp(mode, "file"))
    return;

  size_t weights_bytes, embedding_bytes;
  ws->nseg = weights_plan(m, ws->seg, &weights_bytes, &embedding_bytes);
  size_t total = 0;
  for (int i = 0; i < ws->nseg; i++)
    total += ws->seg[i].bytes;
  size_t budget = placement_hbm_budget(); /* parsed even when unused, so a bad value never goes unnoticed */
  Nodes nodes = choose_nodes(mode, embedding_bytes > 0);
  if (!strcmp(mode, "hbm") && nodes.weights_node >= 0 && weights_bytes > budget)
    fail("weights need %zu bytes of HBM but LLAMA_HBM_BUDGET_GIB allows %zu", weights_bytes, budget);
  if (weights_bytes)
    reserve(&ws->region[0], weights_bytes, nodes.weights_node, ROLE_WEIGHTS, mode);
  if (embedding_bytes)
    reserve(&ws->region[1], embedding_bytes, nodes.embedding_node, ROLE_EMBEDDING, mode);

  double start = monotonic_seconds();
  CopyJob job = {.fd = fd, .ws = ws, .total = total,
                 .chunk = env_size("LLAMA_LOADER_CHUNK", COPY_CHUNK_BYTES),
                 .max_read = env_size("LLAMA_LOADER_MAX_READ", (size_t)1 << 30)};
  Team *team = pool_team(0);
  int workers = team ? team_size(team) : 1;
  job.status = calloc((size_t)workers, sizeof(int));
  if (!job.status)
    fail("cannot allocate loader state");
  if (team)
    team_run(team, copy_slice, &job);
  else
    copy_slice(&job, 0, 1);
  for (int w = 0; w < workers; w++)
    if (job.status[w]) {
      int status = job.status[w];
      free(job.status);
      weights_close(ws);
      if (status < 0)
        fail("unexpected end of file while copying the checkpoint (expected %zu bytes)", file_bytes);
      fail("cannot read the checkpoint: %s", strerror(status));
    }
  free(job.status);
  for (int i = 0; i < 2; i++)
    if (ws->region[i].mapping_base) {
      region_mark_populated(&ws->region[i]);
      region_protect(&ws->region[i]);
      ws->owned_bytes += ws->region[i].logical_bytes;
    }
  metrics.placement_populate_s = monotonic_seconds() - start;

  start = monotonic_seconds();
  for (int i = 0; i < 2; i++)
    if (ws->region[i].mapping_base)
      verify_region(&ws->region[i], mode);
  metrics.placement_verify_s = monotonic_seconds() - start;
  metrics.placement_owned = 1;

  /* The copy read the file with pread, so the mapping was never touched; the views no longer need it. */
  munmap(*mapped, file_bytes);
  *mapped = MAP_FAILED;
  ws->mapped = NULL;
}

void weights_close(WeightStore *ws) {
  for (int i = 0; i < 2; i++)
    region_release(&ws->region[i]);
  ws->nseg = 0;
}
