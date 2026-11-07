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
      for (int g = 0; g < sh->cols / t->group_size; g++) {
        size_t dst = (size_t)(sh->row0 + r) * groups + sh->col0 / t->group_size + g,
               src = (size_t)r * (sh->cols / t->group_size) + g;
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
      assert(contained(r, sh->s, (size_t)sh->rows * (sh->cols / sh->src->group_size) * 4) &&
             (uintptr_t)sh->s % 64 == 0);
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
    size_t row_bytes = (size_t)c->config.hidden_dim + (size_t)c->config.hidden_dim / c->gs * 4;
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
        memcmp(sh->s + (size_t)r * (sh->cols / t->group_size),
               file + t->scale_offset + src_row * row_scales + (size_t)(sh->col0 / t->group_size) * sizeof(float),
               (size_t)(sh->cols / t->group_size) * sizeof(float)))
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
  for (int quantized = 0; quantized < 2; quantized++)
    for (int gs = 2; gs <= 8; gs *= 2) {
      if (!quantized && gs > 2)
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
                c.scratch_limit = 7 * (32 + 32 / (size_t)gs * 4);
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
  printf("shards: %d reconstruction cases ok\n", count);
  return 0;
}
