/* Shard planning and packing (B1): every original tensor must be rebuilt bit for bit from its shards.
 * Random checkpoint bytes stand in for weights, since shards only move bytes. */
#include "shards.h"
#include "loader.h"
#include "numa.h"
#include "runtime.h"
#include "workers.h"
#include <assert.h>
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

int main(void) {
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
