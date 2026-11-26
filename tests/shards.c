/* Shard planning and packing (B1): every original tensor must be rebuilt bit for bit from its shards.
 * Random checkpoint bytes stand in for weights, since shards only move bytes.
 *   test-shards                                          synthetic reconstruction and rejection cases
 *   test-shards verify FILE fp32|int8 [ddr|hbm|file] [THREADS]
 *       shards a real checkpoint over two sockets (the machine's topology, or LLAMA_TOPOLOGY=virtual:2xT), compares
 *       every shard with the file bitwise, verifies the pages of each region, and prints one JSON object */
#include "shards.h"
#include "gemm.h"
#include "loader.h"
#include "numa.h"
#include "runtime.h"
#include "workers.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { HEADER = 28 };

static Config make_config(int dim, int hidden, int layers, int heads, int kv_heads, int vocab, int shared) {
  Config c = {.dim = dim,
              .hidden_dim = hidden,
              .n_layers = layers,
              .n_heads = heads,
              .n_kv_heads = kv_heads,
              .vocab_size = vocab,
              .seq_len = 16,
              .format = 2,
              .max_seq_len = 16,
              .shared = shared,
              .head_dim = dim / heads};
  return c;
}

static unsigned rng_state = 12345;
static unsigned next_random(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

static int random_file(size_t bytes) {
  FILE *f = tmpfile();
  assert(f);
  unsigned char *data = malloc(bytes);
  for (size_t i = 0; i < bytes; i++)
    data[i] = (unsigned char)(next_random() >> 11);
  assert(fwrite(data, 1, bytes, f) == bytes && !fflush(f));
  free(data);
  return dup(fileno(f)); /* the FILE stays open and leaks: this is a test */
}

typedef struct {
  int quantized, gs, shared, sockets, threads;
  Config config;
  size_t scratch_limit;
} Case;

static void *area(NumaRegion *r, size_t bytes, int socket, RegionRole role) {
  if (!bytes)
    return NULL;
  region_reserve(r, bytes, socket, -1, role, 0);
  return r->data;
}

static int contained(const NumaRegion *r, const void *p, size_t bytes) {
  return r->data && (const char *)p >= (const char *)r->data &&
         (const char *)p + bytes <= (const char *)r->data + r->data_bytes;
}

/* Scales per row of a shard and the first of them in the source row: its groups, or for per-channel INT8 (the group is
 * the whole row) the row's one scale, which a column shard keeps whole. */
static int shard_groups(const Shard *sh) {
  return sh->src->group_size == sh->src->cols ? 1 : sh->cols / sh->src->group_size;
}
static int shard_group0(const Shard *sh) {
  return sh->src->group_size == sh->src->cols ? 0 : sh->col0 / sh->src->group_size;
}

/* Rebuilds the tensor from its shards and compares values and scales with the file. */
static void reconstruct(const ShardSet *set, const TensorDesc *t, int fd) {
  int q8 = t->dtype == DT_Q8;
  size_t elem = q8 ? 1 : sizeof(float), groups = q8 ? (size_t)(t->cols / t->group_size) : 0;
  size_t vbytes = (size_t)t->rows * t->cols * elem, sbytes = (size_t)t->rows * groups * sizeof(float);
  unsigned char *value = calloc(vbytes, 1), *scale = calloc(sbytes + 1, 1), *file_v = malloc(vbytes),
                *file_s = malloc(sbytes + 1);
  int *vcover = calloc((size_t)t->rows * t->cols, sizeof(int)),
      *scover = calloc((size_t)t->rows * groups + 1, sizeof(int));
  assert(pread(fd, file_v, vbytes, (off_t)t->offset) == (ssize_t)vbytes);
  if (q8)
    assert(pread(fd, file_s, sbytes, (off_t)t->scale_offset) == (ssize_t)sbytes);
  int shards = 0;
  for (int i = 0; i < set->count; i++) {
    const Shard *sh = &set->t[i];
    if (sh->src->offset != t->offset || sh->src->role != t->role || sh->src->layer != t->layer)
      continue; /* a tied classifier shares the embedding's offset but not its role */
    shards++;
    assert(sh->ld == (size_t)sh->cols && sh->q && (!q8 || sh->s));
    for (int r = 0; r < sh->rows; r++)
      for (int c = 0; c < sh->cols; c++) {
        size_t dst = (size_t)(sh->row0 + r) * t->cols + sh->col0 + c, src = (size_t)r * sh->ld + c;
        memcpy(value + dst * elem, (const char *)sh->q + src * elem, elem);
        vcover[dst]++;
      }
    for (int r = 0; q8 && r < sh->rows; r++)
      for (int g = 0; g < shard_groups(sh); g++) {
        size_t dst = (size_t)(sh->row0 + r) * groups + (size_t)shard_group0(sh) + g,
               src = (size_t)r * (size_t)shard_groups(sh) + g;
        memcpy(scale + dst * sizeof(float), sh->s + src, sizeof(float));
        scover[dst]++;
      }
  }
  assert(shards > 0);
  for (size_t i = 0; i < (size_t)t->rows * t->cols; i++)
    assert(vcover[i] >= 1 && (vcover[i] == 1 || t->rows == 1 || t->role == T_EMBED) &&
           !memcmp(value + i * elem, file_v + i * elem, elem));
  for (size_t i = 0; q8 && i < (size_t)t->rows * groups; i++)
    assert(scover[i] >= 1 && !memcmp(scale + i * 4, file_s + i * 4, 4));
  free(value), free(scale), free(file_v), free(file_s), free(vcover), free(scover);
}

/* C4 packing on a B1 shard: the adapter's job packs the shard's own copy (stride = cols) to the same bytes as packing the
 * matching slice of the source tensor in place (stride = the tensor's cols). BF16 for FP32 tensors, int8 for Q8_0. */
static void check_packing(const ShardSet *set, int fd) {
  for (int i = 0; i < set->count; i++) {
    const Shard *sh = &set->t[i];
    if (sh->kind != SHARD_ROWS && sh->kind != SHARD_COLS)
      continue;
    const TensorDesc *t = sh->src;
    int q8 = t->dtype == DT_Q8;
    size_t elem = q8 ? 1 : sizeof(float), vbytes = (size_t)t->rows * t->cols * elem;
    unsigned char *file_v = malloc(vbytes);
    assert(pread(fd, file_v, vbytes, (off_t)t->offset) == (ssize_t)vbytes);
    const unsigned char *slice = file_v + ((size_t)sh->row0 * t->cols + sh->col0) * elem;
    if (q8) {
      size_t bytes = s8_vnni_bytes(sh->rows, sh->cols);
      int8_t *via_shard = calloc(bytes, 1), *via_slice = calloc(bytes, 1);
      PackS8 job = pack_s8_job_from_shard(via_shard, sh);
      assert(job.rows == sh->rows && job.cols == sh->cols && job.ld == sh->ld);
      pack_s8_vnni_many(&job, 1);
      pack_s8_vnni(via_slice, (const int8_t *)slice, sh->rows, sh->cols, (size_t)t->cols);
      assert(!memcmp(via_shard, via_slice, bytes));
      free(via_shard), free(via_slice);
    } else {
      size_t elements = bf16_vnni_elements(sh->rows, sh->cols);
      uint16_t *via_shard = calloc(elements, sizeof(uint16_t)), *via_slice = calloc(elements, sizeof(uint16_t));
      float *aligned = malloc(vbytes); /* the file bytes are not float-aligned in general */
      memcpy(aligned, file_v, vbytes);
      PackBf16 job = pack_bf16_job_from_shard(via_shard, sh);
      assert(job.rows == sh->rows && job.cols == sh->cols && job.ld == sh->ld);
      pack_bf16_vnni_many(&job, 1);
      pack_bf16_vnni(via_slice, aligned + (size_t)sh->row0 * t->cols + sh->col0, sh->rows, sh->cols, (size_t)t->cols);
      assert(!memcmp(via_shard, via_slice, elements * sizeof(uint16_t)));
      free(via_shard), free(via_slice), free(aligned);
    }
    free(file_v);
  }
}

static void run_case(const Case *c) {
  TensorManifest m;
  manifest_build(&m, &c->config, c->quantized, c->gs, HEADER);
  int fd = random_file(m.file_bytes);
  ShardSet set;
  shards_plan(&set, &m, &c->config, c->sockets, c->gs);
  if (c->scratch_limit)
    set.scratch_limit = c->scratch_limit;
  NumaRegion region[2] = {{0}, {0}}, global = {0};
  void *base[2];
  for (int s = 0; s < c->sockets; s++)
    base[s] = area(&region[s], set.region_bytes[s], s, ROLE_WEIGHTS);
  void *global_base = area(&global, set.global_bytes, 0, ROLE_EMBEDDING);
  if (c->sockets == 1)
    base[1] = NULL;
  shards_bind(&set, base, global_base);

  /* Ownership: every destination lies in its socket's region, aligned, and nothing overlaps. */
  for (int i = 0; i < set.count; i++) {
    const Shard *sh = &set.t[i];
    const NumaRegion *r = sh->kind == SHARD_GLOBAL ? &global : &region[sh->socket];
    size_t elem = sh->src->dtype == DT_Q8 ? 1 : sizeof(float);
    assert(contained(r, sh->q, (size_t)sh->rows * sh->cols * elem) && (uintptr_t)sh->q % 64 == 0);
    if (sh->src->dtype == DT_Q8)
      assert(contained(r, sh->s, (size_t)sh->rows * (size_t)shard_groups(sh) * 4) && (uintptr_t)sh->s % 64 == 0);
    if (sh->kind == SHARD_GLOBAL)
      assert(sh->socket == 0 && sh->src->role == T_EMBED && !c->config.shared);
  }

  PoolConfig pool = {.teams = c->sockets, .threads = {c->threads, c->threads}, .cpus = {NULL, NULL}};
  if (c->threads > 1)
    pool_start(&pool);
  for (int s = 0; s < c->sockets; s++)
    shards_load(&set, fd, s, c->threads > 1 ? pool_team(s) : NULL);
  if (c->threads > 1)
    pool_stop();

  for (int i = 0; i < m.count; i++) {
    const TensorDesc *t = &m.t[i];
    if (t->role == T_ROPE_LEGACY || (t->role == T_EMBED && c->config.shared))
      continue; /* the tied embedding is rebuilt through the classifier alias */
    reconstruct(&set, t, fd);
  }
  check_packing(&set, fd);
  /* COPY shards agree on both sockets; the vocabulary split gives the remainder to the last socket. */
  const Shard *cls0 = shards_find(&set, T_WCLS, -1, 0),
              *cls1 = c->sockets == 2 ? shards_find(&set, T_WCLS, -1, 1) : NULL;
  assert(cls0 && cls0->row0 == 0 && cls0->alias == c->config.shared);
  assert(!cls1 || (cls1->row0 == c->config.vocab_size / 2 && cls1->row0 + cls1->rows == c->config.vocab_size));
  assert(c->sockets == 2 || cls0->rows == c->config.vocab_size);
  if (c->quantized) {
    size_t row_bytes = (size_t)c->config.hidden_dim + (c->gs ? (size_t)c->config.hidden_dim / c->gs : 1) * 4;
    size_t limit = c->scratch_limit ? c->scratch_limit : (size_t)64 << 20;
    for (int s = 0; s < c->sockets; s++)
      assert(set.scratch_peak[s] <= (limit > row_bytes ? limit : row_bytes));
  }
  for (int s = 0; s < c->sockets; s++)
    region_release(&region[s]);
  region_release(&global);
  shards_free(&set);
  manifest_free(&m);
  close(fd);
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
}

/* ---- LLAMA_HYBRID_SPLIT=rows: row-split shards ---------------------------------------------------------------------- */

static size_t aligned64(size_t n) {
  return (n + 63) & ~(size_t)63;
}

/* The shard's row r, wherever it lives. */
static const int8_t *split_row_q(const Shard *sh, int r) {
  int first = sh->rows - sh->ddr_rows;
  return r < first ? (const int8_t *)sh->q + (size_t)r * sh->ld : (const int8_t *)sh->q2 + (size_t)(r - first) * sh->ld;
}
static const float *split_row_s(const Shard *sh, int r) {
  int first = sh->rows - sh->ddr_rows, groups = shard_groups(sh);
  return r < first ? sh->s + (size_t)r * (size_t)groups : sh->s2 + (size_t)(r - first) * (size_t)groups;
}

/* One placement of a Q8 checkpoint with a DDR fraction: the arena sizes are exactly the aligned parts, every row of
 * every shard reads back from the file, and packing the two parts gives the bytes of packing the whole matrix. */
static void run_split_case(int sockets, int threads, int gs, unsigned ppm, int classifier_tier) {
  Config config = make_config(128, 256, 2, 4, 2, 64, 1);
  TensorManifest m;
  manifest_build(&m, &config, 1, gs, HEADER);
  int fd = random_file(m.file_bytes);
  ShardSet set;
  shards_plan(&set, &m, &config, sockets, gs);
  size_t bytes[2][2];
  shards_place(&set, config.n_layers, classifier_tier, 0, ppm, bytes);

  /* Budget arithmetic: each tier's bytes are the sum of the aligned parts placed there, recomputed here independently. */
  for (int s = 0; s < sockets; s++) {
    size_t want[2] = {0, 0};
    for (int i = 0; i < set.count; i++) {
      const Shard *sh = &set.t[i];
      if (sh->socket != s || sh->kind == SHARD_GLOBAL)
        continue;
      int q8 = sh->src->dtype == DT_Q8, ddr = sh->ddr_rows, first = sh->rows - ddr;
      size_t groups = q8 ? (size_t)shard_groups(sh) : 0;
      want[sh->tier] = aligned64(want[sh->tier]) + (size_t)first * (size_t)sh->cols * (q8 ? 1 : 4);
      if (q8)
        want[sh->tier] = aligned64(want[sh->tier]) + (size_t)first * groups * 4;
      if (ddr) {
        want[1] = aligned64(want[1]) + (size_t)ddr * (size_t)sh->cols;
        want[1] = aligned64(want[1]) + (size_t)ddr * groups * 4;
      }
      assert(!ddr || (sh->tier == 0 && sh->kind != SHARD_COPY && sh->src->role != T_WCLS && q8 &&
                      ddr % SHARD_SPLIT_ALIGN == 0 && ddr <= sh->rows * 15 / 16 &&
                      ddr == shards_ddr_rows(sh, ppm)));
    }
    assert(bytes[s][0] == want[0] && bytes[s][1] == want[1]);
  }

  NumaRegion region[2][2] = {{{0}, {0}}, {{0}, {0}}}, global = {0};
  void *base[2][2] = {{NULL, NULL}, {NULL, NULL}};
  for (int s = 0; s < sockets; s++)
    for (int t = 0; t < 2; t++)
      base[s][t] = area(&region[s][t], bytes[s][t], s, ROLE_WEIGHTS);
  void *global_base = area(&global, set.global_bytes, 0, ROLE_EMBEDDING);
  /* The arenas are the plan's: every row-split shard has both places bound inside them, distinct and aligned. */
  shards_bind_placed(&set, base, global_base);
  int split = 0;
  for (int i = 0; i < set.count; i++) {
    const Shard *sh = &set.t[i];
    if (sh->kind == SHARD_GLOBAL || !sh->ddr_rows)
      continue;
    split++;
    int first = sh->rows - sh->ddr_rows;
    assert(contained(&region[sh->socket][0], sh->q, (size_t)first * sh->cols) &&
           contained(&region[sh->socket][0], sh->s, (size_t)first * (size_t)shard_groups(sh) * 4) &&
           contained(&region[sh->socket][1], sh->q2, (size_t)sh->ddr_rows * sh->cols) &&
           contained(&region[sh->socket][1], sh->s2, (size_t)sh->ddr_rows * (size_t)shard_groups(sh) * 4));
    assert((uintptr_t)sh->q2 % 64 == 0 && (uintptr_t)sh->s2 % 64 == 0);
  }
  assert(ppm ? split > 0 : split == 0);

  PoolConfig pool = {.teams = sockets, .threads = {threads, threads}, .cpus = {NULL, NULL}};
  if (threads > 1)
    pool_start(&pool);
  for (int s = 0; s < sockets; s++)
    shards_load_placed(&set, fd, s, threads > 1 ? pool_team(s) : NULL, NULL);
  if (threads > 1)
    pool_stop();

  for (int i = 0; i < set.count; i++) {
    const Shard *sh = &set.t[i];
    if (sh->kind == SHARD_GLOBAL || sh->kind == SHARD_COPY)
      continue;
    const TensorDesc *t = sh->src;
    size_t row_values = (size_t)t->cols, groups = (size_t)(t->cols / t->group_size);
    for (int r = 0; r < sh->rows; r++) {
      size_t src_row = (size_t)(sh->row0 + r);
      unsigned char want_q[1024], want_s[256];
      assert((size_t)sh->cols <= sizeof(want_q) && (size_t)shard_groups(sh) * 4 <= sizeof(want_s));
      assert(pread(fd, want_q, (size_t)sh->cols, (off_t)(t->offset + src_row * row_values + (size_t)sh->col0)) ==
             (ssize_t)sh->cols);
      assert(pread(fd, want_s, (size_t)shard_groups(sh) * 4,
                   (off_t)(t->scale_offset + src_row * groups * 4 + (size_t)shard_group0(sh) * 4)) ==
             (ssize_t)((size_t)shard_groups(sh) * 4));
      assert(!memcmp(split_row_q(sh, r), want_q, (size_t)sh->cols));
      assert(!memcmp(split_row_s(sh, r), want_s, (size_t)shard_groups(sh) * 4));
    }
    /* packing: the two jobs write the bytes of packing the matrix whole (and its transposed scales the same) */
    if (sh->kind == SHARD_ROWS || sh->kind == SHARD_COLS) {
      size_t pbytes = s8_vnni_bytes(sh->rows, sh->cols);
      int8_t *via_jobs = calloc(pbytes, 1), *whole = calloc(pbytes, 1), *rows_copy = malloc((size_t)sh->rows * sh->cols);
      for (int r = 0; r < sh->rows; r++)
        memcpy(rows_copy + (size_t)r * sh->cols, split_row_q(sh, r), (size_t)sh->cols);
      PackS8 jobs[2];
      int n = pack_s8_jobs_from_shard(via_jobs, sh, jobs);
      assert(n == (sh->ddr_rows ? 2 : 1));
      int groups_per_row = shard_groups(sh);
      size_t tbytes = s8_scales_t_bytes(sh->rows, sh->cols, sh->cols / groups_per_row);
      float *st_jobs = calloc(tbytes / 4 + 1, 4), *st_whole = calloc(tbytes / 4 + 1, 4),
            *dense = malloc((size_t)sh->rows * (size_t)groups_per_row * 4);
      for (int r = 0; r < sh->rows; r++)
        memcpy(dense + (size_t)r * (size_t)groups_per_row, split_row_s(sh, r), (size_t)groups_per_row * 4);
      size_t first = (size_t)(sh->rows - sh->ddr_rows);
      for (int j = 0; j < n; j++) {
        jobs[j].st = j ? st_jobs + first / 32 * (size_t)groups_per_row * 32 : st_jobs;
        jobs[j].s = j ? sh->s2 : sh->s;
        jobs[j].groups = groups_per_row;
      }
      pack_s8_vnni_many(jobs, n);
      pack_s8_vnni(whole, rows_copy, sh->rows, sh->cols, (size_t)sh->cols);
      pack_s8_scales_t(st_whole, dense, sh->rows, groups_per_row, 0, (sh->rows + 15) / 16);
      assert(!memcmp(via_jobs, whole, pbytes) && !memcmp(st_jobs, st_whole, tbytes));
      free(via_jobs), free(whole), free(rows_copy), free(st_jobs), free(st_whole), free(dense);
    }
  }
  for (int s = 0; s < sockets; s++)
    for (int t = 0; t < 2; t++)
      region_release(&region[s][t]);
  region_release(&global);
  shards_free(&set);
  manifest_free(&m);
  close(fd);
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
}

/* The rule that picks the rows: only Q8 shards of the seven large matrices, rounded down to whole 32-row steps, at most 15/16
 * of the rows, none for a zero fraction; and the DDR bytes grow with the fraction while the HBM bytes fall. */
static void test_split_rule(void) {
  Config config = make_config(128, 256, 2, 4, 2, 64, 1);
  TensorManifest m;
  manifest_build(&m, &config, 1, 32, HEADER);
  ShardSet set;
  shards_plan(&set, &m, &config, 2, 32);
  size_t previous[2] = {SIZE_MAX, 0}, bytes[2][2];
  for (unsigned ppm = 0; ppm <= 900000; ppm += 100000) {
    shards_place(&set, config.n_layers, 0, 0, ppm, bytes);
    assert(bytes[0][0] <= previous[0] && bytes[0][1] >= previous[1]);
    previous[0] = bytes[0][0], previous[1] = bytes[0][1];
    for (int i = 0; i < set.count; i++) {
      const Shard *sh = &set.t[i];
      int rows = sh->ddr_rows;
      if (sh->kind == SHARD_COPY || sh->kind == SHARD_GLOBAL || sh->src->role == T_WCLS) {
        assert(!rows);
        continue;
      }
      assert(rows == shards_ddr_rows(sh, ppm) && rows % 32 == 0 && rows <= sh->rows * 15 / 16);
      assert(rows <= (int)((long long)sh->rows * ppm / 1000000));
      assert(!ppm || (long long)sh->rows * ppm / 1000000 - rows < 32 || rows == (sh->rows * 15 / 16) / 32 * 32);
    }
  }
  shards_place(&set, config.n_layers, 0, 0, 0, bytes);
  size_t whole[2] = {bytes[0][0], bytes[0][1]};
  shards_place(&set, config.n_layers, 0, 0, 500000, bytes);
  /* splitting moves bytes, up to 4 lines of alignment slack per split shard (two parts, values and scales) */
  assert(bytes[0][0] + bytes[0][1] >= whole[0] - 64 * 1000 && bytes[0][1] > whole[1]);
  shards_place(&set, config.n_layers, 0, 0, 0, bytes);
  shards_free(&set);
  manifest_free(&m);
  /* FP32 shards are never split */
  manifest_build(&m, &config, 0, 0, HEADER);
  shards_plan(&set, &m, &config, 2, 0);
  shards_place(&set, config.n_layers, 0, 0, 900000, bytes);
  for (int i = 0; i < set.count; i++)
    assert(!set.t[i].ddr_rows);
  shards_free(&set);
  manifest_free(&m);
}

static void expect_rejected(const Config *p, int quantized, int gs, int sockets, int plan_gs, const char *text) {
  TensorManifest m;
  manifest_build(&m, p, quantized, gs, HEADER);
  ShardSet set;
  char err[512];
  int status = shards_plan_try(&set, &m, p, sockets, plan_gs, err, sizeof(err));
  if (status != -1 || !strstr(err, text)) {
    fprintf(stderr, "expected rejection containing '%s', got status %d '%s'\n", text, status, status ? err : "");
    abort();
  }
  manifest_free(&m);
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
}

/* ---- verify: a real checkpoint, two sockets -------------------------------------------------------------------------
 */

typedef struct {
  ShardSet *set;
  int fd, socket;
  Team *team;
} LoadArg;

static void *load_main(void *p) {
  LoadArg *arg = p;
  shards_load(arg->set, arg->fd, arg->socket, arg->team);
  return NULL;
}

static int shard_matches(const Shard *sh, const unsigned char *file) {
  const TensorDesc *t = sh->src;
  size_t elem = t->dtype == DT_Q8 ? 1 : sizeof(float), row_values = (size_t)t->cols * elem;
  size_t groups = t->dtype == DT_Q8 ? (size_t)(t->cols / t->group_size) : 0, row_scales = groups * sizeof(float);
  for (int r = 0; r < sh->rows; r++) {
    size_t src_row = (size_t)(sh->row0 + r);
    if (memcmp((const char *)sh->q + (size_t)r * sh->ld * elem,
               file + t->offset + src_row * row_values + (size_t)sh->col0 * elem, (size_t)sh->cols * elem))
      return 0;
    if (t->dtype == DT_Q8 &&
        memcmp(sh->s + (size_t)r * (size_t)shard_groups(sh),
               file + t->scale_offset + src_row * row_scales + (size_t)shard_group0(sh) * sizeof(float),
               (size_t)shard_groups(sh) * sizeof(float)))
      return 0;
  }
  return 1;
}

static int verify_mode(int argc, char **argv) {
  if (argc < 4)
    return 2;
  int quantized = !strcmp(argv[3], "int8");
  const char *placement = argc > 4 ? argv[4] : "file";
  Config config;
  TensorManifest m;
  int shared, gs, fd;
  float *data;
  size_t bytes;
  checkpoint_map(argv[2], quantized, &config, &shared, &gs, &fd, &data, &bytes, &m);
  ShardSet set;
  shards_plan(&set, &m, &config, 2, gs);
  const Topology *topology = numa_topology();
  char err[256];
  if (numa_check_placement(topology, 2, placement, err, sizeof(err)))
    fail("%s", err);
  PoolConfig pool = {.teams = 2};
  for (int s = 0; s < 2; s++) {
    pool.threads[s] = argc > 5 ? atoi(argv[5]) : topology->socket[s].ncpus;
    pool.cpus[s] = pool.threads[s] <= topology->socket[s].ncpus ? topology->socket[s].cpus : NULL;
  }
  pool_start(&pool);
  NumaRegion region[2] = {{0}, {0}}, global = {0};
  void *base[2];
  for (int s = 0; s < 2; s++) {
    int node = !strcmp(placement, "hbm")   ? topology->socket[s].hbm_node
               : !strcmp(placement, "ddr") ? topology->socket[s].ddr_node
                                           : -1;
    region_reserve(&region[s], set.region_bytes[s], s, node, ROLE_WEIGHTS, region_thp_default());
    base[s] = region[s].data;
  }
  if (set.global_bytes)
    region_reserve(&global, set.global_bytes, 0, strcmp(placement, "file") ? topology->socket[0].ddr_node : -1,
                   ROLE_EMBEDDING, region_thp_default());
  shards_bind(&set, base, global.data);
  LoadArg arg[2] = {{&set, fd, 0, pool_team(0)}, {&set, fd, 1, pool_team(1)}};
  double start = monotonic_seconds();
  pthread_t thread;
  assert(!pthread_create(&thread, NULL, load_main, &arg[1]));
  load_main(&arg[0]);
  pthread_join(thread, NULL);
  double load_seconds = monotonic_seconds() - start;

  int match = 1;
  for (int i = 0; i < set.count; i++)
    match &= shard_matches(&set.t[i], (const unsigned char *)data);
  int placed = 1;
  printf("{\"sockets\": 2, \"precision\": \"%s\", \"gs\": %d, \"shared\": %d, \"shards\": %d, \"match\": %d, "
         "\"load_seconds\": %.4f, \"rows_seconds\": [%.4f, %.4f], \"cols_seconds\": [%.4f, %.4f], "
         "\"scratch_peak\": [%zu, %zu], \"region_bytes\": [%zu, %zu], \"global_bytes\": %zu, \"regions\": [",
         quantized ? "int8" : "fp32", gs, shared, set.count, match, load_seconds, set.rows_seconds[0],
         set.rows_seconds[1], set.cols_seconds[0], set.cols_seconds[1], set.scratch_peak[0], set.scratch_peak[1],
         set.region_bytes[0], set.region_bytes[1], set.global_bytes);
  NumaRegion *all[3] = {&region[0], &region[1], &global};
  for (int i = 0; i < 3; i++) {
    PageCheck check = {0};
    int status = all[i]->node >= 0 ? region_verify(all[i], 0, &check) : -1;
    if (status == 0 && (check.wrong_node || check.missing || check.errors))
      placed = 0;
    printf("%s{\"socket\": %d, \"node\": %d, \"verified\": %d, \"queried\": %zu, \"on_node\": %zu, "
           "\"wrong_node\": %zu, \"missing\": %zu}",
           i ? ", " : "", all[i]->socket, all[i]->node, status == 0, check.queried, check.on_node, check.wrong_node,
           check.missing);
  }
  printf("], \"placement_ok\": %d}\n", placed);
  for (int i = 0; i < 3; i++)
    region_release(all[i]);
  pool_stop();
  shards_free(&set);
  manifest_free(&m);
  return match && placed ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc >= 2 && !strcmp(argv[1], "verify"))
    return verify_mode(argc, argv);
  int count = 0;
  const int group_sizes[] = {2, 4, 8, 0}; /* 0: per-channel INT8, one scale per row */
  for (int quantized = 0; quantized < 2; quantized++)
    for (size_t gi = 0; gi < sizeof(group_sizes) / sizeof(*group_sizes); gi++) {
      int gs = group_sizes[gi];
      if (!quantized && gs != 2)
        continue;
      for (int shared = 0; shared < 2; shared++)
        for (int sockets = 1; sockets <= 2; sockets++)
          for (int threads = 1; threads <= 7; threads += threads < 3 ? 2 : 4) {
            for (int vocab = 37; vocab <= 128; vocab += 91) { /* odd and even */
              Case c = {.quantized = quantized,
                        .gs = quantized ? gs : 0,
                        .shared = shared,
                        .sockets = sockets,
                        .threads = threads,
                        .config = make_config(16, 32, 2, 4, 2, vocab, shared)};
              run_case(&c);
              if (quantized) { /* a block of 7 source rows */
                c.scratch_limit = 7 * (32 + (gs ? 32 / (size_t)gs : 1) * 4);
                run_case(&c);
              }
              count++;
            }
          }
    }
  /* A larger GQA ratio and an uneven head count per socket. */
  Case big = {
      .quantized = 1, .gs = 4, .shared = 1, .sockets = 2, .threads = 3, .config = make_config(48, 96, 1, 6, 2, 53, 1)};
  run_case(&big);

  Config ok = make_config(16, 32, 2, 4, 2, 37, 0);
  /* A group size that divides dim and hidden (the engine's own check) but not their per-socket halves. */
  Config odd_half = make_config(24, 48, 2, 6, 2, 37, 0);
  expect_rejected(&odd_half, 1, 8, 2, 8, "group size 8 does not divide"); /* dim/S = 12 */
  {
    TensorManifest mo;
    manifest_build(&mo, &odd_half, 1, 8, HEADER);
    ShardSet so;
    char eo[256];
    assert(shards_plan_try(&so, &mo, &odd_half, 1, 8, eo, sizeof(eo)) == 0); /* one socket needs no halves */
    shards_free(&so);
    manifest_free(&mo);
  }
  expect_rejected(&ok, 1, 4, 2, 8, "does not match the checkpoint's 4");
  Config one_kv = make_config(4, 8, 2, 2, 1, 128010, 1); /* the existing fixture */
  expect_rejected(&one_kv, 0, 0, 2, 0, "n_kv_heads=1");
  Config odd_hidden = make_config(16, 33, 2, 4, 2, 37, 0);
  expect_rejected(&odd_hidden, 0, 0, 2, 0, "hidden_dim=33");
  Config kv3 = make_config(24, 48, 2, 6, 3, 37, 0);
  expect_rejected(&kv3, 0, 0, 2, 0, "n_kv_heads=3");
  expect_rejected(&ok, 0, 0, 3, 0, "only 1 or 2");
  Config groups_split = make_config(16, 32, 2, 4, 4, 37, 0); /* 4 KV heads per 4 heads is fine */
  TensorManifest m;
  manifest_build(&m, &groups_split, 0, 0, HEADER);
  ShardSet set;
  char err[256];
  assert(shards_plan_try(&set, &m, &groups_split, 2, 0, err, sizeof(err)) == 0);
  shards_free(&set);
  manifest_free(&m);
  test_split_rule();
  for (int sockets = 1; sockets <= 2; sockets++)
    for (int threads = 1; threads <= 3; threads += 2)
      for (int gs = 32; gs <= 64; gs *= 2)
        for (int step = 0; step < 3; step++)
          for (int cls = 0; cls < 2; cls++) {
            run_split_case(sockets, threads, gs, step == 0 ? 0u : step == 1 ? 500000u : 900000u, cls);
            count++;
          }
  printf("shards: %d reconstruction cases ok\n", count);
  return 0;
}
