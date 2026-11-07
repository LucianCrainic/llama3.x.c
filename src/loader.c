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

/* HBM bytes this process has claimed against LLAMA_HBM_BUDGET_GIB: weights, KV caches and the work region. */
static size_t hbm_claimed;

static size_t budget_claim(const char *mode, int node, size_t bytes, const char *what) {
  size_t budget = placement_hbm_budget(); /* parsed even when unused, so a bad value never goes unnoticed */
  if (strcmp(mode, "hbm") || node < 0)
    return 0;
  if (bytes > budget || hbm_claimed > budget - bytes)
    fail("%s need %zu bytes of HBM but LLAMA_HBM_BUDGET_GIB leaves %zu of %zu", what, bytes,
         budget > hbm_claimed ? budget - hbm_claimed : 0, budget);
  hbm_claimed += bytes;
  return bytes;
}

static void budget_release(size_t *claimed) {
  hbm_claimed -= *claimed;
  *claimed = 0;
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
  Nodes nodes = choose_nodes(mode, embedding_bytes > 0);
  ws->budget_claimed = budget_claim(mode, nodes.weights_node, weights_bytes, "the weights");
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
  budget_release(&ws->budget_claimed);
  ws->nseg = 0;
}

/* ---- run state (A5) ----------------------------------------------------------------------------------------------- */

void runmem_init(RunMemory *rm, size_t kv_bytes_each) {
  const char *mode = placement_mode();
  *rm = (RunMemory){.mode = mode, .node = -1, .kv_bytes = kv_bytes_each};
  placement_hbm_budget();
  if (!strcmp(mode, "file"))
    return;
  rm->phase = RUNMEM_SIZING;
  rm->node = choose_nodes(mode, 0).weights_node;
  /* The caches are reserved now; the budget also counts the work region once it is sized (runmem_seal). */
  rm->budget_claimed = budget_claim(mode, rm->node, checked_mul(2, kv_bytes_each), "the KV caches");
  reserve(&rm->kv[0], kv_bytes_each, rm->node, ROLE_KV_K, mode);
  reserve(&rm->kv[1], kv_bytes_each, rm->node, ROLE_KV_V, mode);
  /* Reserved, not resident: accounting follows the extent, residency follows the first write at each position. */
  region_mark_populated(&rm->kv[0]);
  region_mark_populated(&rm->kv[1]);
}

int runmem_sizing(const RunMemory *rm) {
  return rm->phase == RUNMEM_SIZING;
}

void *runmem_alloc(RunMemory *rm, size_t count, size_t size) {
  if (rm->phase == RUNMEM_HEAP)
    return memory_alloc(count, size, MEM_ACTIVATION);
  size_t bytes = checked_mul(count, size), at = checked_add(rm->used, (size_t)(CACHE_LINE - 1)) / CACHE_LINE * CACHE_LINE;
  rm->used = checked_add(at, bytes);
  return rm->phase == RUNMEM_SIZING ? NULL : (char *)rm->work.data + at;
}

void runmem_attention(RunMemory *rm, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len,
                      int rows) {
  if (rm->phase == RUNMEM_HEAP) {
    k_attention_init_rows(work, n_heads, n_kv_heads, head_dim, seq_len, rows, NULL);
    return;
  }
  void *memory = runmem_alloc(rm, k_attention_bytes_rows(n_heads, n_kv_heads, head_dim, seq_len, rows), 1);
  if (!runmem_sizing(rm))
    k_attention_init_rows(work, n_heads, n_kv_heads, head_dim, seq_len, rows, memory);
}

static void zero_slice(const void *job, int worker, int workers) {
  const NumaRegion *r = job;
  size_t lo = (size_t)((unsigned long long)r->logical_bytes * (unsigned)worker / (unsigned)workers);
  size_t hi = (size_t)((unsigned long long)r->logical_bytes * (unsigned)(worker + 1) / (unsigned)workers);
  memset((char *)r->data + lo, 0, hi - lo);
}

void runmem_seal(RunMemory *rm) {
  if (rm->phase != RUNMEM_SIZING)
    return;
  size_t bytes = rm->used ? rm->used : 1;
  rm->budget_claimed += budget_claim(rm->mode, rm->node, bytes, "the run-state buffers");
  reserve(&rm->work, bytes, rm->node, ROLE_WORK, rm->mode);
  /* The owning team writes the region once, so its pages exist on the socket that uses them. */
  Team *team = pool_team(0);
  if (team)
    team_run(team, zero_slice, &rm->work);
  else
    zero_slice(&rm->work, 0, 1);
  region_mark_populated(&rm->work);
  rm->used = 0; /* the second allocation pass starts over */
  rm->phase = RUNMEM_CARVE;
}

float *runmem_cache(RunMemory *rm, int which) {
  if (rm->phase == RUNMEM_HEAP)
    return memory_alloc(rm->kv_bytes / sizeof(float), sizeof(float), MEM_KV);
  return rm->kv[which].data;
}

static int inside(const NumaRegion *r, const void *p) {
  return r->mapping_base && (const char *)p >= (const char *)r->data && (const char *)p < (const char *)r->data + r->data_bytes;
}

void runmem_free(RunMemory *rm, void *pointer) {
  /* Region buffers die with their region; anything else (heap mode, the lazy batch logits) is a plain allocation. */
  if (inside(&rm->work, pointer) || inside(&rm->kv[0], pointer) || inside(&rm->kv[1], pointer))
    return;
  memory_free(pointer);
}

void runmem_release(RunMemory *rm) {
  region_release(&rm->work);
  region_release(&rm->kv[0]);
  region_release(&rm->kv[1]);
  budget_release(&rm->budget_claimed);
  rm->phase = RUNMEM_HEAP;
}

void runmem_kv_stats(const RunMemory *rm, size_t active_bytes, int verify, KvStats *out) {
  *out = (KvStats){.reserved_bytes = checked_mul(2, rm->kv_bytes), .active_bytes = active_bytes, .node = -1};
  if (rm->phase == RUNMEM_HEAP || rm->kv[0].node < 0)
    return;
  out->node = rm->kv[0].node;
  if (!verify)
    return;
  size_t page = (size_t)sysconf(_SC_PAGESIZE);
  out->full = 1;
  double resident = 0;
  for (int i = 0; i < 2; i++) {
    PageCheck check = {0};
    if (region_verify(&rm->kv[i], region_verify_samples_default(), &check) != 0) {
      if (placement_strict(rm->mode))
        fail("page placement of the KV caches on node %d cannot be verified on this system", rm->kv[i].node);
      *out = (KvStats){.reserved_bytes = out->reserved_bytes, .active_bytes = active_bytes, .node = out->node};
      return;
    }
    size_t thp = 0;
    if (region_thp_bytes(&rm->kv[i], &thp) != 0)
      thp = 0;
    region_record_verification(&rm->kv[i], &check, thp);
    out->full &= check.full;
    out->pages_queried += check.queried;
    out->pages_wrong_node += check.wrong_node;
    out->pages_missing += check.missing;
    size_t pages = (rm->kv[i].logical_bytes + page - 1) / page, present = check.queried - check.missing - check.errors;
    resident += check.queried ? (double)present / (double)check.queried * (double)pages * (double)page : 0;
    if ((check.wrong_node || check.errors) && placement_strict(rm->mode))
      fail("KV cache: %zu of %zu queried pages are not on node %d (%zu elsewhere, %zu errors)", check.queried - check.on_node - check.missing,
           check.queried, rm->kv[i].node, check.wrong_node, check.errors);
  }
  out->verified = 1;
  out->resident_bytes = (size_t)(resident + 0.5);
}
