/* The manifest must reproduce, byte for byte, the layout the old pointer-bumping
 * loaders produced. The oracle below is that arithmetic, kept here only.
 *   test-manifest                    synthetic configs, no files
 *   test-manifest fp32|int8 <file>   checkpoint_map + manifest against the oracle */
#include "loader.h"
#include "runtime.h"
#include "shards.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* Old run.c memory_map_weights, in bytes from the start of the weights. */
typedef struct {
  size_t embed, rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3, rms_final, wcls;
  size_t end;
} Oracle;

static Oracle oracle_f32(const Config *p, int shared) {
  size_t head_size = p->dim / p->n_heads, layers = p->n_layers, d = p->dim, f = sizeof(float);
  size_t ptr = 0;
  Oracle o = {0};
  o.embed = ptr;
  ptr += (size_t)p->vocab_size * d * f;
  o.rms_att = ptr;
  ptr += layers * d * f;
  o.wq = ptr;
  ptr += layers * d * (p->n_heads * head_size) * f;
  o.wk = ptr;
  ptr += layers * d * (p->n_kv_heads * head_size) * f;
  o.wv = ptr;
  ptr += layers * d * (p->n_kv_heads * head_size) * f;
  o.wo = ptr;
  ptr += layers * (p->n_heads * head_size) * d * f;
  o.rms_ffn = ptr;
  ptr += layers * d * f;
  o.w1 = ptr;
  ptr += layers * d * p->hidden_dim * f;
  o.w2 = ptr;
  ptr += layers * (size_t)p->hidden_dim * d * f;
  o.w3 = ptr;
  ptr += layers * d * p->hidden_dim * f;
  o.rms_final = ptr;
  ptr += d * f;
  if (p->format == 0)
    ptr += (size_t)p->seq_len * head_size * f;
  o.wcls = shared ? o.embed : ptr;
  o.end = ptr + (shared ? 0 : (size_t)p->vocab_size * d * f);
  return o;
}

/* Old runq.c layout: norms, then each quantized array as values followed by scales per tensor.
 * q[i] and s[i] hold per-layer offsets for the arrays that have layers. */
typedef struct {
  size_t rms_att, rms_ffn, rms_final;
  size_t q_embed, s_embed, q_cls, s_cls;
  size_t q[T_ROLE_COUNT][64], s[T_ROLE_COUNT][64];
  size_t end;
} OracleQ8;

static void oracle_q8_array(OracleQ8 *o, size_t *ptr, TensorRole role, int n, size_t size_each, int gs) {
  for (int i = 0; i < n; i++) {
    o->q[role][i] = *ptr;
    *ptr += size_each;
    o->s[role][i] = *ptr;
    *ptr += size_each / gs * sizeof(float);
  }
}

static OracleQ8 oracle_q8(const Config *p, int shared, int gs) {
  size_t head_size = p->dim / p->n_heads, d = p->dim, ptr = 0;
  OracleQ8 o = {0};
  o.rms_att = ptr;
  ptr += (size_t)p->n_layers * d * sizeof(float);
  o.rms_ffn = ptr;
  ptr += (size_t)p->n_layers * d * sizeof(float);
  o.rms_final = ptr;
  ptr += d * sizeof(float);
  oracle_q8_array(&o, &ptr, T_EMBED, 1, (size_t)p->vocab_size * d, gs);
  oracle_q8_array(&o, &ptr, T_WQ, p->n_layers, d * (p->n_heads * head_size), gs);
  oracle_q8_array(&o, &ptr, T_WK, p->n_layers, d * (p->n_kv_heads * head_size), gs);
  oracle_q8_array(&o, &ptr, T_WV, p->n_layers, d * (p->n_kv_heads * head_size), gs);
  oracle_q8_array(&o, &ptr, T_WO, p->n_layers, (p->n_heads * head_size) * d, gs);
  oracle_q8_array(&o, &ptr, T_W1, p->n_layers, d * p->hidden_dim, gs);
  oracle_q8_array(&o, &ptr, T_W2, p->n_layers, (size_t)p->hidden_dim * d, gs);
  oracle_q8_array(&o, &ptr, T_W3, p->n_layers, d * p->hidden_dim, gs);
  if (!shared)
    oracle_q8_array(&o, &ptr, T_WCLS, 1, d * p->vocab_size, gs);
  o.end = ptr;
  return o;
}

/* Every manifest tensor is addressable by (role, layer) and the non-alias tensors
 * tile the file after the header with no gap or overlap. */
static void check_structure(const TensorManifest *m, const Config *p, size_t header, int shared) {
  size_t cursor = header;
  for (int i = 0; i < m->count; i++) {
    const TensorDesc *t = &m->t[i];
    assert(manifest_find(m, t->role, t->layer) == t);
    if (t->alias_of >= 0) {
      assert(t->role == T_WCLS && shared && t->alias_of < i);
      const TensorDesc *target = &m->t[t->alias_of];
      assert(target->role == T_EMBED && t->offset == target->offset && t->bytes == target->bytes);
      continue;
    }
    assert(t->offset == cursor);
    cursor += t->bytes;
    if (t->dtype == DT_F32)
      assert(t->scale_offset == 0 && t->group_size == 0);
    else
      assert(t->scale_offset == t->offset + (size_t)t->rows * t->cols && t->group_size > 0);
  }
  assert(cursor == m->file_bytes);
  assert(!!manifest_find(m, T_ROPE_LEGACY, -1) == (p->format == 0));
  assert(!manifest_find(m, T_WQ, p->n_layers) && !manifest_find(m, T_WQ, -1));
  assert((manifest_find(m, T_WCLS, -1)->alias_of >= 0) == !!shared);
}

static void compare_f32(const TensorManifest *m, const Config *p, int shared, size_t header) {
  Oracle o = oracle_f32(p, shared);
  size_t d = p->dim, kv = (size_t)p->head_dim * p->n_kv_heads, h = p->hidden_dim, f = sizeof(float);
  struct {
    TensorRole role;
    size_t base, size;
  } layered[] = {{T_RMS_ATT, o.rms_att, d * f}, {T_WQ, o.wq, d * d * f}, {T_WK, o.wk, kv * d * f},
                 {T_WV, o.wv, kv * d * f},      {T_WO, o.wo, d * d * f}, {T_RMS_FFN, o.rms_ffn, d * f},
                 {T_W1, o.w1, h * d * f},       {T_W2, o.w2, h * d * f}, {T_W3, o.w3, h * d * f}};
  for (size_t r = 0; r < sizeof(layered) / sizeof(*layered); r++)
    for (int l = 0; l < p->n_layers; l++) {
      const TensorDesc *t = manifest_find(m, layered[r].role, l);
      assert(t && t->offset == header + layered[r].base + (size_t)l * layered[r].size);
      assert(t->bytes == layered[r].size && t->dtype == DT_F32);
    }
  assert(manifest_find(m, T_EMBED, -1)->offset == header + o.embed);
  assert(manifest_find(m, T_RMS_FINAL, -1)->offset == header + o.rms_final);
  assert(manifest_find(m, T_WCLS, -1)->offset == header + o.wcls);
  assert(m->file_bytes == header + o.end);
  check_structure(m, p, header, shared);
}

static void compare_q8(const TensorManifest *m, const Config *p, int shared, int gs, size_t header) {
  OracleQ8 o = oracle_q8(p, shared, gs);
  size_t d = p->dim;
  assert(manifest_find(m, T_RMS_ATT, 0)->offset == header + o.rms_att);
  assert(manifest_find(m, T_RMS_FFN, 0)->offset == header + o.rms_ffn);
  assert(manifest_find(m, T_RMS_FINAL, -1)->offset == header + o.rms_final);
  TensorRole roles[] = {T_WQ, T_WK, T_WV, T_WO, T_W1, T_W2, T_W3};
  for (size_t r = 0; r < sizeof(roles) / sizeof(*roles); r++)
    for (int l = 0; l < p->n_layers; l++) {
      const TensorDesc *t = manifest_find(m, roles[r], l);
      assert(t && t->dtype == DT_Q8 && t->group_size == gs);
      assert(t->offset == header + o.q[roles[r]][l] && t->scale_offset == header + o.s[roles[r]][l]);
    }
  const TensorDesc *e = manifest_find(m, T_EMBED, -1);
  assert(e->offset == header + o.q[T_EMBED][0] && e->scale_offset == header + o.s[T_EMBED][0]);
  const TensorDesc *c = manifest_find(m, T_WCLS, -1);
  if (shared)
    assert(c->offset == e->offset && c->scale_offset == e->scale_offset);
  else
    assert(c->offset == header + o.q[T_WCLS][0] && c->scale_offset == header + o.s[T_WCLS][0] &&
           c->rows == p->vocab_size && c->cols == (int)d);
  assert(m->file_bytes == header + o.end);
  check_structure(m, p, header, shared);
}

static Config make_config(int format, int shared, int dim, int heads, int kv_heads) {
  Config p = {.dim = dim,
              .hidden_dim = dim * 2,
              .n_layers = 3,
              .n_heads = heads,
              .n_kv_heads = kv_heads,
              .vocab_size = 20,
              .seq_len = 12,
              .format = format,
              .max_seq_len = 12,
              .shared = shared,
              .head_dim = dim / heads};
  return p;
}

/* A4 layout: segments tile the non-alias tensors exactly, in file order, per class, keeping the offset modulo 64. */
static void check_plan(const TensorManifest *m, int quantized, int shared) {
  WeightSegment seg[WEIGHT_SEGMENTS];
  size_t weights, embedding;
  int n = weights_plan(m, seg, &weights, &embedding);
  /* FP32 embedding leads the file; Q8 puts the norms before it, so an untied Q8 embedding splits the weights. */
  assert(n == (shared ? 1 : quantized ? 3 : 2));
  assert((embedding > 0) == !shared);
  size_t covered = 0, last_end[2] = {0, 0}, expected_end = seg[0].src_begin;
  size_t tensors = 0;
  for (int i = 0; i < m->count; i++)
    tensors += m->t[i].alias_of < 0 ? m->t[i].bytes : 0;
  for (int i = 0; i < n; i++) {
    assert(seg[i].src_begin == expected_end);
    expected_end = seg[i].src_begin + seg[i].bytes;
    assert(seg[i].dst_offset % 64 == seg[i].src_begin % 64);
    assert(seg[i].dst_offset >= last_end[seg[i].embedding]);
    last_end[seg[i].embedding] = seg[i].dst_offset + seg[i].bytes;
    covered += seg[i].bytes;
  }
  assert(covered == tensors && expected_end == m->file_bytes);
  assert(weights == last_end[0] && embedding == last_end[1]);
  assert(weights + embedding < tensors + (size_t)n * 64);
}

static void synthetic(void) {
  const int formats[] = {0, 2, 3};
  for (int f = 0; f < 3; f++)
    for (int shared = 0; shared < 2; shared++)
      for (int shape = 0; shape < 2; shape++) {
        Config p = shape ? make_config(formats[f], shared, 16, 4, 2) : make_config(formats[f], shared, 4, 2, 1);
        size_t header = formats[f] == 0 ? 28 : 256;
        TensorManifest m;
        manifest_build(&m, &p, 0, 0, header);
        compare_f32(&m, &p, shared, header);
        check_plan(&m, 0, shared);
        manifest_free(&m);
        if (formats[f] == 0)
          continue; /* Q8 exists only as ak42 versions 2 and 3 */
        int gs = shape ? 8 : 2;
        manifest_build(&m, &p, 1, gs, header);
        compare_q8(&m, &p, shared, gs, header);
        check_plan(&m, 1, shared);
        manifest_free(&m);
      }
  assert(!manifest_find(&(TensorManifest){0}, T_EMBED, -1));
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
}

static void checkpoint(int quantized, const char *path) {
  Config p;
  int shared, gs, fd;
  float *data;
  size_t bytes;
  TensorManifest m;
  checkpoint_map(path, quantized, &p, &shared, &gs, &fd, &data, &bytes, &m);
  size_t header = p.format == 0 ? 28 : 256;
  assert(m.file_bytes == bytes || p.format != 3);
  assert(m.file_bytes <= bytes);
  if (quantized)
    compare_q8(&m, &p, shared, gs, header);
  else
    compare_f32(&m, &p, shared, header);
  printf("manifest: %s format %d %s: %d tensors, %zu bytes\n", quantized ? "int8" : "fp32", p.format,
         shared ? "shared" : "unshared", m.count, m.file_bytes);
  manifest_free(&m);
  munmap(data, bytes);
  close(fd);
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
}

int main(int argc, char **argv) {
  if (argc == 1) {
    synthetic();
    return 0;
  }
  if (argc != 3 || (strcmp(argv[1], "fp32") && strcmp(argv[1], "int8")))
    return 2;
  checkpoint(!strcmp(argv[1], "int8"), argv[2]);
  return 0;
}
