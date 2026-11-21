/* Attention kernels: split-K decode (split, group), batched causal prefill attention (split, tiled), the variant
 * dispatch (LLAMA_ATTN, LLAMA_ATTN_DECODE) and the K/V cache layout (LLAMA_KV_LAYOUT). Moved out of kernels_simd.c
 * unchanged (F8). */
#include "kernels_internal.h"
#include "runtime.h"
#include "testing.h"
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* Rows of one (kv group, chunk) task of the split batched attention. Rows of a block are computed one after the other,
 * each exactly as without the block, so the result is bitwise independent of it (E0 swept 1..16; 4 kept, F5). */
enum { BATCH_ROW_BLOCK = 4 };
/* Tiled attention: query vectors per task (rows of a block x heads of the group). E2 swept 1..16 and the claim order
 * (plan order against longest first); 16 in plan order was kept, the knobs were removed (F5). */
enum { TILED_QUERIES = 16 };

// ----------------------------------------------------------------------------
// Split-K partials

/* The partial of a chunk that holds no position of the row (tiled, E2): zero value sum, the lowest maximum, zero sum of
 * exponentials. A real chunk always has sum >= 1 (its maximum contributes exp(0)), so the test cannot match one. */
static int partial_empty(const float *chunk, int head_dim) {
  return chunk[head_dim + 1] == 0.0f && chunk[head_dim] == -FLT_MAX;
}

#if HAVE_AVX512
static void partial_write_empty(float *o, int head_dim) {
  memset(o, 0, (size_t)head_dim * sizeof(float));
  o[head_dim] = -FLT_MAX;
  o[head_dim + 1] = 0.0f;
}

/* One GQA group over positions [t0, t1): scores, local max/sum and the
 * unnormalized value sum for each query head. K/V chunks are at most 256
 * positions, so heads after the first re-read them from L2, not memory. */
AVX512 static void attention_chunk_avx512(const float *q, const float *kc, const float *vc, int kv_dim, int head_dim,
                                          int heads, int t0, int t1, float *scores, size_t score_stride,
                                          float *partial, size_t head_stride) {
  enum { MAX_VECTORS = 16 };
  int vectors = (head_dim + 15) / 16;
  __mmask16 last = tail_mask(head_dim - (vectors - 1) * 16);
  float divisor = sqrtf(head_dim);
  const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)), finite = _mm512_set1_ps(FLT_MAX);
  for (int h = 0; h < heads; h++) {
    __m512 qv[MAX_VECTORS];
    for (int v = 0; v < vectors; v++)
      qv[v] = _mm512_maskz_loadu_ps(v + 1 == vectors ? last : 0xFFFF, q + (size_t)h * head_dim + v * 16);
    float *a = scores + (size_t)h * score_stride; /* a[t - t0] is the score of position t */
    int t = t0;
    for (; t + 4 <= t1; t += 4) {
      const float *k0 = kc + (size_t)t * kv_dim, *k1 = k0 + kv_dim, *k2 = k1 + kv_dim, *k3 = k2 + kv_dim;
      __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
      for (int v = 0; v < vectors; v++) {
        __mmask16 m = v + 1 == vectors ? last : 0xFFFF;
        s0 = _mm512_fmadd_ps(qv[v], _mm512_maskz_loadu_ps(m, k0 + v * 16), s0);
        s1 = _mm512_fmadd_ps(qv[v], _mm512_maskz_loadu_ps(m, k1 + v * 16), s1);
        s2 = _mm512_fmadd_ps(qv[v], _mm512_maskz_loadu_ps(m, k2 + v * 16), s2);
        s3 = _mm512_fmadd_ps(qv[v], _mm512_maskz_loadu_ps(m, k3 + v * 16), s3);
      }
      a[t - t0] = _mm512_reduce_add_ps(s0) / divisor;
      a[t - t0 + 1] = _mm512_reduce_add_ps(s1) / divisor;
      a[t - t0 + 2] = _mm512_reduce_add_ps(s2) / divisor;
      a[t - t0 + 3] = _mm512_reduce_add_ps(s3) / divisor;
    }
    for (; t < t1; t++) {
      const float *k0 = kc + (size_t)t * kv_dim;
      __m512 s0 = _mm512_setzero_ps();
      for (int v = 0; v < vectors; v++)
        s0 = _mm512_fmadd_ps(qv[v], _mm512_maskz_loadu_ps(v + 1 == vectors ? last : 0xFFFF, k0 + v * 16), s0);
      a[t - t0] = _mm512_reduce_add_ps(s0) / divisor;
    }
    __m512 vmax = _mm512_set1_ps(-FLT_MAX);
    for (t = t0; t < t1; t += 16) {
      __mmask16 m = tail_mask(t1 - t);
      __m512 s = _mm512_maskz_loadu_ps(m, a + (t - t0));
      if (_mm512_mask_cmp_ps_mask(m, _mm512_and_ps(s, abs_mask), finite, _CMP_LE_OQ) != m)
        fail("nonfinite value in softmax");
      vmax = _mm512_mask_max_ps(vmax, m, vmax, s);
    }
    float max_val = _mm512_reduce_max_ps(vmax);
    __m512 vsum = _mm512_setzero_ps(), shift = _mm512_set1_ps(max_val);
    for (t = t0; t < t1; t += 16) {
      __mmask16 m = tail_mask(t1 - t);
      __m512 e = exp512(_mm512_sub_ps(_mm512_maskz_loadu_ps(m, a + (t - t0)), shift));
      _mm512_mask_storeu_ps(a + (t - t0), m, e);
      vsum = _mm512_mask_add_ps(vsum, m, vsum, e);
    }
    __m512 ov[MAX_VECTORS];
    for (int v = 0; v < vectors; v++)
      ov[v] = _mm512_setzero_ps();
    for (t = t0; t < t1; t++) {
      const float *vr = vc + (size_t)t * kv_dim;
      __m512 p = _mm512_set1_ps(a[t - t0]);
      for (int v = 0; v < vectors; v++)
        ov[v] = _mm512_fmadd_ps(p, _mm512_maskz_loadu_ps(v + 1 == vectors ? last : 0xFFFF, vr + v * 16), ov[v]);
    }
    float *o = partial + (size_t)h * head_stride;
    for (int v = 0; v < vectors; v++)
      _mm512_mask_storeu_ps(o + v * 16, v + 1 == vectors ? last : 0xFFFF, ov[v]);
    o[head_dim] = max_val;
    o[head_dim + 1] = _mm512_reduce_add_ps(vsum);
  }
}

/* ---- E2: tiled batched attention -------------------------------------------------------------------------------------
 * One task = one KV group, one chunk [t0, t1) of positions and a block of `nrows` consecutive rows (row r sees the
 * positions below vis0 + r). The chunk is the same for every row of the block; a row sees only its own prefix of it, and
 * a row with no position in the chunk writes an empty partial. The block's query vectors are its rows x the group's
 * heads (at most 16); K is read once per tile of 32 positions for all of them. */
enum { TILED_POS = 32, TILED_MAX_Q = 16, TILED_MAX_CHUNK = 256, TILED_QG = 8 };

AVX512 static inline __mmask16 lane_mask(int count) {
  return count <= 0 ? (__mmask16)0 : count >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << count) - 1u);
}

/* In-place transpose of a 16 x 16 tile held in 16 registers: 64 shuffles (unpack ps, unpack pd, two f32x4 shuffles). */
AVX512 static inline void transpose16(__m512 *r) {
  __m512 t[16], u[16];
  for (int i = 0; i < 8; i++) {
    t[2 * i] = _mm512_unpacklo_ps(r[2 * i], r[2 * i + 1]);
    t[2 * i + 1] = _mm512_unpackhi_ps(r[2 * i], r[2 * i + 1]);
  }
  for (int g = 0; g < 4; g++) {
    __m512d a = _mm512_castps_pd(t[4 * g]), b = _mm512_castps_pd(t[4 * g + 1]), c = _mm512_castps_pd(t[4 * g + 2]),
            d = _mm512_castps_pd(t[4 * g + 3]);
    u[4 * g] = _mm512_castpd_ps(_mm512_unpacklo_pd(a, c));
    u[4 * g + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(a, c));
    u[4 * g + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(b, d));
    u[4 * g + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(b, d));
  }
  for (int k = 0; k < 4; k++) {
    __m512 a0 = _mm512_shuffle_f32x4(u[k], u[4 + k], 0x88), a1 = _mm512_shuffle_f32x4(u[k], u[4 + k], 0xDD);
    __m512 b0 = _mm512_shuffle_f32x4(u[8 + k], u[12 + k], 0x88), b1 = _mm512_shuffle_f32x4(u[8 + k], u[12 + k], 0xDD);
    r[k] = _mm512_shuffle_f32x4(a0, b0, 0x88);
    r[4 + k] = _mm512_shuffle_f32x4(a1, b1, 0x88);
    r[8 + k] = _mm512_shuffle_f32x4(a0, b0, 0xDD);
    r[12 + k] = _mm512_shuffle_f32x4(a1, b1, 0xDD);
  }
}

/* Keys of positions t..t+cnt-1 (cnt <= 32) of one KV head as kt[d][position], so a score tile is a broadcast-FMA over d
 * with no horizontal reduction. Positions at and after cnt are zero; rows past them are never read. */
AVX512 static void tiled_transpose_keys(const float *kc, int kv_dim, int head_dim, int t, int cnt, float *kt) {
  int halves = cnt > 16 ? 2 : 1;
  for (int d0 = 0; d0 < head_dim; d0 += 16) {
    __mmask16 dm = lane_mask(head_dim - d0);
    int dn = head_dim - d0 < 16 ? head_dim - d0 : 16;
    for (int h = 0; h < halves; h++) {
      __m512 r[16];
      for (int p = 0; p < 16; p++)
        r[p] = h * 16 + p < cnt ? _mm512_maskz_loadu_ps(dm, kc + (size_t)(t + h * 16 + p) * kv_dim + d0)
                                : _mm512_setzero_ps();
      transpose16(r);
      for (int i = 0; i < dn; i++)
        _mm512_storeu_ps(kt + (size_t)(d0 + i) * TILED_POS + h * 16, r[i]);
    }
  }
}

/* Scores of TILED_QG query vectors (qd[d * 16 + j] is element d of vector j, already scaled: the 16 vectors of the task
 * are interleaved per d so the broadcast operands of one d sit in one cache line) against the transposed tile: 8 x 2
 * accumulators, per d two loads of kt and eight broadcast operands for 16 FMAs. `nv` is 1 or 2 position vectors. */
AVX512 static inline void tiled_scores8(const float *kt, const float *qd, int head_dim, int nv, float *out,
                                        size_t ostride, int cnt) {
  __m512 a[TILED_QG][2];
  for (int j = 0; j < TILED_QG; j++)
    a[j][0] = a[j][1] = _mm512_setzero_ps();
  if (nv == 2) {
    for (int d = 0; d < head_dim; d++, kt += TILED_POS, qd += TILED_MAX_Q) {
      __m512 k0 = _mm512_loadu_ps(kt), k1 = _mm512_loadu_ps(kt + 16);
      for (int j = 0; j < TILED_QG; j++) {
        __m512 b = _mm512_set1_ps(qd[j]);
        a[j][0] = _mm512_fmadd_ps(b, k0, a[j][0]);
        a[j][1] = _mm512_fmadd_ps(b, k1, a[j][1]);
      }
    }
  } else {
    for (int d = 0; d < head_dim; d++, kt += TILED_POS, qd += TILED_MAX_Q) {
      __m512 k0 = _mm512_loadu_ps(kt);
      for (int j = 0; j < TILED_QG; j++)
        a[j][0] = _mm512_fmadd_ps(_mm512_set1_ps(qd[j]), k0, a[j][0]);
    }
  }
  for (int j = 0; j < TILED_QG; j++) {
    _mm512_mask_storeu_ps(out + j * ostride, lane_mask(cnt), a[j][0]);
    if (nv == 2)
      _mm512_mask_storeu_ps(out + j * ostride + 16, lane_mask(cnt - 16), a[j][1]);
  }
}

/* P x V for TILED_QG query vectors and one slice of `dv` (1 or 2) vectors of the head: the accumulators come from the
 * task's output scratch (rescaled by alpha when the running maximum moved), one V row load feeds 8 x dv FMAs whose
 * broadcast operand is a probability in L1, and they go back to the scratch. 16 accumulators, two V registers. */
AVX512 static inline __attribute__((always_inline)) void tiled_pv(const float *p8, const float *v, int kv_dim, int npos,
                                                                   float *o8, size_t ostride, const float *alpha8,
                                                                   __mmask16 m0, __mmask16 m1, int dv) {
  __m512 a[TILED_QG][2];
  for (int j = 0; j < TILED_QG; j++) {
    a[j][0] = _mm512_loadu_ps(o8 + j * ostride);
    a[j][1] = dv == 2 ? _mm512_loadu_ps(o8 + j * ostride + 16) : _mm512_setzero_ps();
    if (alpha8) {
      __m512 alpha = _mm512_set1_ps(alpha8[j]);
      a[j][0] = _mm512_mul_ps(a[j][0], alpha);
      if (dv == 2)
        a[j][1] = _mm512_mul_ps(a[j][1], alpha);
    }
  }
  for (int pos = 0; pos < npos; pos++) {
    const float *vr = v + (size_t)pos * kv_dim;
    __m512 v0 = _mm512_maskz_loadu_ps(m0, vr), v1 = dv == 2 ? _mm512_maskz_loadu_ps(m1, vr + 16) : _mm512_setzero_ps();
    for (int j = 0; j < TILED_QG; j++) {
      __m512 b = _mm512_set1_ps(p8[j * TILED_POS + pos]);
      a[j][0] = _mm512_fmadd_ps(b, v0, a[j][0]);
      if (dv == 2)
        a[j][1] = _mm512_fmadd_ps(b, v1, a[j][1]);
    }
  }
  for (int j = 0; j < TILED_QG; j++) {
    _mm512_storeu_ps(o8 + j * ostride, a[j][0]);
    if (dv == 2)
      _mm512_storeu_ps(o8 + j * ostride + 16, a[j][1]);
  }
}

/* The task. Per tile of 32 positions: transpose K once; then per group of 8 query vectors scores (registers to an 8 x 32
 * L1 tile), online softmax (running maximum m and sum l per query vector; when the maximum moves, the accumulator and l
 * are rescaled by exp(m_old - m_new)), probabilities to an L1 tile, and PV into the output scratch. Nothing but K, V
 * and Q is read from beyond L1, and no score is ever written to memory the task does not own. Positions a row may not
 * see get probability exactly zero (the lanes are masked, not given a large negative score). */
AVX512 static void attention_tiled_avx512(const float *q, size_t q_row_stride, int nrows, int kv_mul, int head_dim,
                                          int vis0, int t0, int t1, const float *kc, const float *vc, int kv_dim,
                                          float *partial, size_t row_stride, size_t head_stride) {
  _Alignas(64) float qs[TILED_MAX_Q * 256], kt[256 * TILED_POS], os[TILED_MAX_Q * 256], ps[TILED_MAX_Q * TILED_POS],
      sc[TILED_QG * TILED_POS];
  float m[TILED_MAX_Q], l[TILED_MAX_Q], alpha[TILED_MAX_Q], lane[16];
  int vectors = (head_dim + 15) / 16, hdp = vectors * 16, nq = nrows * kv_mul;
  int nqp = (nq + TILED_QG - 1) / TILED_QG * TILED_QG;
  __mmask16 last = lane_mask(head_dim - (vectors - 1) * 16);
  /* Q for the block, scaled by 1/sqrt(head_dim) once (no division per score) and transposed to qs[d][query vector], with
   * zero vectors up to 16 (a 16 x 16 transpose per 16 elements of the head). */
  const __m512 scale = _mm512_set1_ps(1.0f / sqrtf((float)head_dim));
  for (int d0 = 0; d0 < hdp; d0 += 16) {
    __mmask16 dm = lane_mask(head_dim - d0);
    __m512 r[16];
    for (int qi = 0; qi < TILED_MAX_Q; qi++)
      r[qi] = qi < nq ? _mm512_mul_ps(_mm512_maskz_loadu_ps(dm, q + (size_t)(qi / kv_mul) * q_row_stride +
                                                                    (size_t)(qi % kv_mul) * head_dim + d0),
                                      scale)
                      : _mm512_setzero_ps();
    transpose16(r);
    for (int c = 0; c < 16; c++)
      _mm512_storeu_ps(qs + (size_t)(d0 + c) * TILED_MAX_Q, r[c]);
  }
  for (int qi = 0; qi < nqp; qi++)
    for (int v = 0; v < vectors; v++)
      _mm512_storeu_ps(os + (size_t)qi * hdp + v * 16, _mm512_setzero_ps());
  for (int qi = 0; qi < TILED_MAX_Q; qi++) {
    m[qi] = -FLT_MAX, l[qi] = 0.0f, alpha[qi] = 1.0f;
    for (int i = 0; i < TILED_POS; i += 16)
      _mm512_storeu_ps(ps + qi * TILED_POS + i, _mm512_setzero_ps());
  }
  const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)), finite = _mm512_set1_ps(FLT_MAX);
  int tend = t1 < vis0 + nrows - 1 ? t1 : vis0 + nrows - 1; /* beyond the last row's prefix nothing is read */
  for (int t = t0; t < tend; t += TILED_POS) {
    int cnt = tend - t < TILED_POS ? tend - t : TILED_POS;
    tiled_transpose_keys(kc, kv_dim, head_dim, t, cnt, kt);
    for (int q0 = 0; q0 < nqp; q0 += TILED_QG) {
      /* the last real query vector of the group sees the most positions of this tile */
      int qlast = q0 + TILED_QG - 1 < nq ? q0 + TILED_QG - 1 : nq - 1, rlast = qlast / kv_mul;
      int gn = (vis0 + rlast < t + cnt ? vis0 + rlast : t + cnt) - t;
      if (gn <= 0)
        continue; /* no query vector of the group sees any position of this tile: nothing to add */
      tiled_scores8(kt, qs + q0, head_dim, gn > 16 ? 2 : 1, sc, TILED_POS, gn);
      int rescale = 0;
      for (int j = 0; j < TILED_QG; j++) {
        int qi = q0 + j, n = qi < nq ? (vis0 + qi / kv_mul < t + cnt ? vis0 + qi / kv_mul : t + cnt) - t : 0;
        float *p = ps + (size_t)qi * TILED_POS;
        alpha[qi] = 1.0f;
        if (n <= 0) { /* a row above the diagonal of this tile, or a padding row */
          _mm512_storeu_ps(p, _mm512_setzero_ps());
          _mm512_storeu_ps(p + 16, _mm512_setzero_ps());
          continue;
        }
        __mmask16 k0 = lane_mask(n), k1 = lane_mask(n - 16);
        __m512 s0 = _mm512_maskz_loadu_ps(k0, sc + j * TILED_POS), s1 = _mm512_maskz_loadu_ps(k1, sc + j * TILED_POS + 16);
        if ((_mm512_mask_cmp_ps_mask(k0, _mm512_and_ps(s0, abs_mask), finite, _CMP_LE_OQ) != k0) ||
            (_mm512_mask_cmp_ps_mask(k1, _mm512_and_ps(s1, abs_mask), finite, _CMP_LE_OQ) != k1))
          fail("nonfinite value in softmax");
        __m512 vmax = _mm512_set1_ps(-FLT_MAX);
        vmax = _mm512_mask_max_ps(vmax, k0, vmax, s0);
        vmax = _mm512_mask_max_ps(vmax, k1, vmax, s1);
        float tile_max = _mm512_reduce_max_ps(vmax), old = m[qi];
        if (tile_max > old) { /* the maximum moved: what is accumulated so far is scaled by exp(old - new) */
          _mm512_storeu_ps(lane, exp512(_mm512_set1_ps(old - tile_max)));
          alpha[qi] = lane[0];
          l[qi] *= lane[0];
          m[qi] = tile_max;
          rescale = 1;
        }
        __m512 shift = _mm512_set1_ps(m[qi]);
        __m512 e0 = _mm512_maskz_mov_ps(k0, exp512(_mm512_sub_ps(s0, shift)));
        __m512 e1 = _mm512_maskz_mov_ps(k1, exp512(_mm512_sub_ps(s1, shift)));
        _mm512_storeu_ps(p, e0);
        _mm512_storeu_ps(p + 16, e1);
        l[qi] += _mm512_reduce_add_ps(_mm512_add_ps(e0, e1));
      }
      for (int s = 0; s < vectors; s += 2) { /* slices of two vectors of the head */
        __mmask16 m0 = s + 1 == vectors ? last : 0xFFFF, m1 = s + 2 == vectors ? last : 0xFFFF;
        const float *pq = ps + (size_t)q0 * TILED_POS, *v = vc + (size_t)t * kv_dim + s * 16;
        float *o8 = os + (size_t)q0 * hdp + s * 16;
        if (s + 1 < vectors)
          tiled_pv(pq, v, kv_dim, gn, o8, (size_t)hdp, rescale ? alpha + q0 : NULL, m0, m1, 2);
        else
          tiled_pv(pq, v, kv_dim, gn, o8, (size_t)hdp, rescale ? alpha + q0 : NULL, m0, m1, 1);
      }
    }
  }
  for (int qi = 0; qi < nq; qi++) {
    int r = qi / kv_mul, h = qi % kv_mul, end = t1 < vis0 + r ? t1 : vis0 + r;
    float *o = partial + (size_t)r * row_stride + (size_t)h * head_stride;
    if (end <= t0) {
      partial_write_empty(o, head_dim);
      continue;
    }
    for (int v = 0; v < vectors; v++)
      _mm512_mask_storeu_ps(o + v * 16, v + 1 == vectors ? last : 0xFFFF,
                            _mm512_loadu_ps(os + (size_t)qi * hdp + v * 16));
    o[head_dim] = m[qi];
    o[head_dim + 1] = l[qi];
  }
}

// ----------------------------------------------------------------------------
// Group decode attention (E5)

/* Single-row attention for one KV group over one chunk [t0, t1) of positions: the group's `nq` query heads (at most 16)
 * are handled together, so each K and V row is read from memory once per group. Scores of 16 positions are computed
 * as 16 partial vectors (one per position, accumulated over the head with the pre-scaled Q) and reduced by one 16-way
 * transpose-add: no horizontal reduction and no division per score. Softmax is online per tile of 32 positions (running
 * maximum m and sum l per head, accumulator and l rescaled by exp(m_old - m_new)) and PV is fused into the same pass.
 * The result is a split-K partial [acc(head_dim), max, sum] per head, merged by merge_chunks like the split kernel's.
 */
enum { GROUP_POS = 32, GROUP_MAX_Q = 16, GROUP_MAX_HD = 256 };

/* Lane j of the result is the sum of the 16 lanes of a[j]: 30 shuffles and 15 adds for 16 sums (a full transpose takes
 * 64 shuffles). Levels: float unpack within 128-bit blocks, double unpack, then the four 128-bit blocks. */
AVX512 static inline __m512 group_reduce16(const __m512 *a) {
  __m512 s[8], u[4];
  for (int i = 0; i < 8; i++)
    s[i] = _mm512_add_ps(_mm512_unpacklo_ps(a[2 * i], a[2 * i + 1]), _mm512_unpackhi_ps(a[2 * i], a[2 * i + 1]));
  for (int i = 0; i < 4; i++) {
    __m512d x = _mm512_castps_pd(s[2 * i]), y = _mm512_castps_pd(s[2 * i + 1]);
    u[i] = _mm512_add_ps(_mm512_castpd_ps(_mm512_unpacklo_pd(x, y)), _mm512_castpd_ps(_mm512_unpackhi_pd(x, y)));
  }
  __m512 t0 = _mm512_add_ps(_mm512_shuffle_f32x4(u[0], u[1], 0x88), _mm512_shuffle_f32x4(u[0], u[1], 0xDD));
  __m512 t1 = _mm512_add_ps(_mm512_shuffle_f32x4(u[2], u[3], 0x88), _mm512_shuffle_f32x4(u[2], u[3], 0xDD));
  return _mm512_add_ps(_mm512_shuffle_f32x4(t0, t1, 0x88), _mm512_shuffle_f32x4(t0, t1, 0xDD));
}

/* Scores (already scaled through q) of 16 positions whose key rows are kp[0..15]; lane p is position p. */
AVX512 static inline __attribute__((always_inline)) __m512 group_scores16(const float *q, const float *const *kp,
                                                                          int vectors, __mmask16 last) {
  __m512 a[16];
  for (int p = 0; p < 16; p++)
    a[p] = _mm512_setzero_ps();
  for (int v = 0; v + 1 < vectors; v++) {
    __m512 qv = _mm512_loadu_ps(q + v * 16);
    for (int p = 0; p < 16; p++)
      a[p] = _mm512_fmadd_ps(qv, _mm512_loadu_ps(kp[p] + v * 16), a[p]);
  }
  __m512 qv = _mm512_loadu_ps(q + (vectors - 1) * 16);
  for (int p = 0; p < 16; p++)
    a[p] = _mm512_fmadd_ps(qv, _mm512_maskz_loadu_ps(last, kp[p] + (vectors - 1) * 16), a[p]);
  return group_reduce16(a);
}

/* P x V for NH query heads and a slice of up to DV vectors of the head (dv of them used): accumulators from the task's
 * output scratch (rescaled by alpha), one V row load feeds NH x dv FMAs with a probability broadcast, and they go back.
 */
AVX512 static inline __attribute__((always_inline)) void group_pv(const float *p, const float *v, int kv_dim, int npos,
                                                                  float *o, size_t ostride, const float *alpha,
                                                                  __mmask16 mlast, int dv, const int NH, const int DV) {
  __m512 a[8][8];
  __mmask16 mk[8];
  for (int x = 0; x < DV; x++)
    mk[x] = x + 1 == dv ? mlast : (__mmask16)0xFFFF;
  for (int j = 0; j < NH; j++) {
    __m512 al = _mm512_set1_ps(alpha[j]);
    for (int x = 0; x < DV; x++)
      a[j][x] = x < dv ? _mm512_mul_ps(_mm512_loadu_ps(o + j * ostride + x * 16), al) : _mm512_setzero_ps();
  }
  for (int pos = 0; pos < npos; pos++) {
    const float *vr = v + (size_t)pos * kv_dim;
    __m512 vv[8];
    for (int x = 0; x < DV; x++)
      vv[x] = x < dv ? _mm512_maskz_loadu_ps(mk[x], vr + x * 16) : _mm512_setzero_ps();
    for (int j = 0; j < NH; j++) {
      __m512 b = _mm512_set1_ps(p[j * GROUP_POS + pos]);
      for (int x = 0; x < DV; x++)
        a[j][x] = _mm512_fmadd_ps(b, vv[x], a[j][x]);
    }
  }
  for (int j = 0; j < NH; j++)
    for (int x = 0; x < DV; x++)
      if (x < dv)
        _mm512_storeu_ps(o + j * ostride + x * 16, a[j][x]);
}

AVX512 static void attention_group_avx512(const float *q, int nq, int head_dim, int t0, int t1, const float *kc,
                                          const float *vc, int kv_dim, float *partial, size_t head_stride) {
  _Alignas(64) float qs[GROUP_MAX_Q * GROUP_MAX_HD], os[GROUP_MAX_Q * GROUP_MAX_HD], ps[GROUP_MAX_Q * GROUP_POS],
      zero[GROUP_MAX_HD];
  float m[GROUP_MAX_Q], l[GROUP_MAX_Q], alpha[GROUP_MAX_Q], lane[16];
  int vectors = (head_dim + 15) / 16, hdp = vectors * 16;
  __mmask16 last = lane_mask(head_dim - (vectors - 1) * 16);
  const __m512 scale = _mm512_set1_ps(1.0f / sqrtf((float)head_dim));
  /* Q scaled once by 1/sqrt(head_dim); the tail of the last vector is zero so it never meets a stray key element. */
  for (int h = 0; h < nq; h++) {
    for (int v = 0; v < vectors; v++) {
      __mmask16 mk = v + 1 == vectors ? last : (__mmask16)0xFFFF;
      _mm512_storeu_ps(qs + (size_t)h * hdp + v * 16,
                       _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, q + (size_t)h * head_dim + v * 16), scale));
      _mm512_storeu_ps(os + (size_t)h * hdp + v * 16, _mm512_setzero_ps());
    }
    m[h] = -FLT_MAX, l[h] = 0.0f, alpha[h] = 1.0f;
  }
  for (int v = 0; v < hdp; v += 16)
    _mm512_storeu_ps(zero + v, _mm512_setzero_ps());
  for (int i = 0; i < GROUP_MAX_Q * GROUP_POS; i += 16)
    _mm512_storeu_ps(ps + i, _mm512_setzero_ps());
  const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)), finite = _mm512_set1_ps(FLT_MAX);
  for (int t = t0; t < t1; t += GROUP_POS) {
    int cnt = t1 - t < GROUP_POS ? t1 - t : GROUP_POS;
    for (int p = 0; p < cnt; p++) /* bring the tile's V rows in while the scores are computed */
      for (int d = 0; d < head_dim; d += 16)
        __builtin_prefetch(vc + (size_t)(t + p) * kv_dim + d, 0, 3);
    const float *kp[2][16];
    for (int p = 0; p < GROUP_POS; p++)
      kp[p / 16][p % 16] = p < cnt ? kc + (size_t)(t + p) * kv_dim : zero;
    __mmask16 k0 = lane_mask(cnt), k1 = lane_mask(cnt - 16);
    for (int h = 0; h < nq; h++) {
      const float *qh = qs + (size_t)h * hdp;
      __m512 s0 = group_scores16(qh, kp[0], vectors, last);
      __m512 s1 = cnt > 16 ? group_scores16(qh, kp[1], vectors, last) : _mm512_setzero_ps();
      if ((_mm512_mask_cmp_ps_mask(k0, _mm512_and_ps(s0, abs_mask), finite, _CMP_LE_OQ) != k0) ||
          (_mm512_mask_cmp_ps_mask(k1, _mm512_and_ps(s1, abs_mask), finite, _CMP_LE_OQ) != k1))
        fail("nonfinite value in softmax");
      __m512 vmax = _mm512_set1_ps(-FLT_MAX);
      vmax = _mm512_mask_max_ps(vmax, k0, vmax, s0);
      vmax = _mm512_mask_max_ps(vmax, k1, vmax, s1);
      float tile_max = _mm512_reduce_max_ps(vmax), old = m[h];
      alpha[h] = 1.0f;      /* 1 when the maximum did not move: one code path serves both cases */
      if (tile_max > old) { /* the maximum moved: what is accumulated so far is scaled by exp(old - new) */
        _mm512_storeu_ps(lane, exp512(_mm512_set1_ps(old - tile_max)));
        alpha[h] = lane[0];
        l[h] *= lane[0];
        m[h] = tile_max;
      }
      __m512 shift = _mm512_set1_ps(m[h]);
      __m512 e0 = _mm512_maskz_mov_ps(k0, exp512(_mm512_sub_ps(s0, shift)));
      __m512 e1 = _mm512_maskz_mov_ps(k1, exp512(_mm512_sub_ps(s1, shift)));
      _mm512_storeu_ps(ps + h * GROUP_POS, e0);
      _mm512_storeu_ps(ps + h * GROUP_POS + 16, e1);
      l[h] += _mm512_reduce_add_ps(_mm512_add_ps(e0, e1));
    }
    for (int h0 = 0; h0 < nq;) {
      int left = nq - h0, nh = left >= 8 ? 8 : left >= 4 ? 4 : left >= 2 ? 2 : 1;
      int dvmax = nh == 8 ? 2 : nh == 4 ? 4 : 8;
      for (int s = 0; s < vectors; s += dvmax) {
        int dv = vectors - s < dvmax ? vectors - s : dvmax;
        __mmask16 ml = s + dv == vectors ? last : (__mmask16)0xFFFF;
        const float *pp = ps + (size_t)h0 * GROUP_POS, *v = vc + (size_t)t * kv_dim + s * 16, *al = alpha + h0;
        float *o = os + (size_t)h0 * hdp + s * 16;
        switch (nh) {
        case 8:
          group_pv(pp, v, kv_dim, cnt, o, (size_t)hdp, al, ml, dv, 8, 2);
          break;
        case 4:
          group_pv(pp, v, kv_dim, cnt, o, (size_t)hdp, al, ml, dv, 4, 4);
          break;
        case 2:
          group_pv(pp, v, kv_dim, cnt, o, (size_t)hdp, al, ml, dv, 2, 8);
          break;
        default:
          group_pv(pp, v, kv_dim, cnt, o, (size_t)hdp, al, ml, dv, 1, 8);
          break;
        }
      }
      h0 += nh;
    }
  }
  for (int h = 0; h < nq; h++) {
    float *o = partial + (size_t)h * head_stride;
    if (t1 <= t0) {
      partial_write_empty(o, head_dim);
      continue;
    }
    for (int v = 0; v < vectors; v++)
      _mm512_mask_storeu_ps(o + v * 16, v + 1 == vectors ? last : (__mmask16)0xFFFF,
                            _mm512_loadu_ps(os + (size_t)h * hdp + v * 16));
    o[head_dim] = m[h];
    o[head_dim + 1] = l[h];
  }
}
#endif

// ----------------------------------------------------------------------------
// Attention


/* Enough chunks to occupy every thread, each of 32..256 positions where the
 * prefix allows, bounded by the workspace. */
static int attention_chunks(int positions, int n_kv_heads, int limit, int workers) {
  int chunks = (workers + n_kv_heads - 1) / n_kv_heads;
  int longest = (positions + 31) / 32, shortest = (positions + 255) / 256;
  if (chunks > longest)
    chunks = longest;
  if (chunks < shortest)
    chunks = shortest;
  if (chunks > limit)
    chunks = limit;
  return chunks < 1 ? 1 : chunks;
}
static int attention_max_chunks(int n_kv_heads, int seq_len, int workers) {
  int by_threads = (workers + n_kv_heads - 1) / n_kv_heads, by_length = (seq_len + 255) / 256;
  return by_threads > by_length ? by_threads : by_length;
}

static size_t attention_row_floats(int n_heads, int head_dim, int max_chunks) {
  return checked_mul(checked_mul((size_t)n_heads, (size_t)max_chunks), (size_t)head_dim + 2);
}

/* The batched attention's partials workspace: 32 MiB (the LLAMA_ATTN_WORKSPACE_MIB knob was removed, F5). Tests lower it
 * with k_attention_set_workspace_mib to run several waves on small shapes; 0 means one row per wave. */
static long attention_workspace_mib = 32;

void k_attention_set_workspace_mib(long mib) {
  attention_workspace_mib = mib < 0 ? 32 : mib;
}

static size_t attention_budget_floats(void) {
  long mib = attention_workspace_mib;
  return mib > 0 ? (size_t)mib * (1u << 20) / sizeof(float) : 1;
}

/* Rows of partials that fit: at least one, at most `rows`, otherwise as many as the budget holds. */
static size_t attention_rows_capacity(int n_heads, int head_dim, int max_chunks, int rows) {
  size_t per_row = attention_row_floats(n_heads, head_dim, max_chunks), fit = attention_budget_floats() / per_row;
  size_t want = rows > 1 ? (size_t)rows : 1;
  return fit < 1 ? 1 : fit < want ? fit : want;
}

size_t k_attention_bytes_rows(int n_heads, int n_kv_heads, int head_dim, int seq_len, int rows) {
  int chunks = attention_max_chunks(n_kv_heads, seq_len, max_threads());
  return checked_mul(checked_mul(attention_rows_capacity(n_heads, head_dim, chunks, rows),
                                 attention_row_floats(n_heads, head_dim, chunks)),
                     sizeof(float));
}

size_t k_attention_bytes(int n_heads, int n_kv_heads, int head_dim, int seq_len) {
  return k_attention_bytes_rows(n_heads, n_kv_heads, head_dim, seq_len, 1);
}

void k_attention_init_rows(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int rows,
                           float *memory) {
  /* The team must be set before this: it sizes the split-K workspace. */
  work->workers = max_threads();
  work->max_chunks = attention_max_chunks(n_kv_heads, seq_len, work->workers);
  work->n_heads = n_heads;
  work->head_dim = head_dim;
  work->external = memory != NULL;
  work->capacity = checked_mul(attention_rows_capacity(n_heads, head_dim, work->max_chunks, rows),
                               attention_row_floats(n_heads, head_dim, work->max_chunks));
  work->partial = memory ? memory : memory_alloc(work->capacity, sizeof(float), MEM_ACTIVATION);
}

void k_attention_init_in(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, float *memory) {
  k_attention_init_rows(work, n_heads, n_kv_heads, head_dim, seq_len, 1, memory);
}

void k_attention_init(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len) {
  k_attention_init_in(work, n_heads, n_kv_heads, head_dim, seq_len, NULL);
}

int k_attention_chunk_count(const AttentionWork *work, int n_kv_heads, int position) {
  return attention_chunks(position + 1, n_kv_heads, work->max_chunks, work->workers);
}

void k_attention_free(AttentionWork *work) {
  if (!work->external)
    memory_free(work->partial);
  work->partial = NULL;
}

typedef struct {
  float *out;
  const float *q, *key_cache, *value_cache;
  float *att;
  int n_heads, n_kv_heads, head_size, seq_len, pos;
  size_t pos_stride, head_base; /* floats between positions of a head and between the bases of KV heads */
} ScalarAttention;

static void attention_scalar_head(const ScalarAttention *j, int h) {
  int head_size = j->head_size, pos = j->pos, kv_mul = j->n_heads / j->n_kv_heads;
  // get the query vector for this head
  const float *qh = j->q + h * head_size;
  // attention scores for this head
  float *a = j->att + (size_t)h * j->seq_len;
  // iterate over all timesteps, including the current one
  for (int t = 0; t <= pos; t++) {
    // get the key vector for this head and at this timestep
    const float *k = j->key_cache + (size_t)t * j->pos_stride + (size_t)(h / kv_mul) * j->head_base;
    // calculate the attention score as the dot product of q and k
    float score = 0.0f;
    for (int i = 0; i < head_size; i++) {
      score += qh[i] * k[i];
    }
    score /= sqrtf(head_size);
    // save the score to the attention buffer
    a[t] = score;
  }

  // softmax the scores to get attention weights, from 0..pos inclusively
  softmax(a, pos + 1);

  // weighted sum of the values, store back into xb
  float *xb = j->out + h * head_size;
  memset(xb, 0, head_size * sizeof(float));
  for (int t = 0; t <= pos; t++) {
    // get the value vector for this head and at this timestep
    const float *v = j->value_cache + (size_t)t * j->pos_stride + (size_t)(h / kv_mul) * j->head_base;
    // get the attention weight for this timestep
    float w = a[t];
    // accumulate the weighted value into xb
    for (int i = 0; i < head_size; i++) {
      xb[i] += w * v[i];
    }
  }
}

static void attention_scalar_rows(const void *p, int h0, int h1) {
  for (int h = h0; h < h1; h++)
    attention_scalar_head(p, h);
}

static void attention_scalar(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                             int n_heads, int n_kv_heads, int head_size, int seq_len, int pos, size_t pos_stride,
                             size_t head_base) {
  ScalarAttention job = {out, q, key_cache, value_cache, att, n_heads, n_kv_heads, head_size, seq_len, pos, pos_stride,
                         head_base};
  if (current_team) {
    team_parallel_rows(current_team, n_heads, attention_scalar_rows, &job);
    return;
  }
  for (int h = 0; h < n_heads; h++)
    attention_scalar_head(&job, h);
}

/* Merge chunk results: rescale each by exp(m_c - M) and normalize. `base` holds `chunks` slots of `slot` floats. Empty
 * partials are skipped, so their value sum is never read. */
static void merge_chunks(const float *base, size_t slot, int chunks, int head_dim, float *o) {
  float max_val = base[head_dim];
  for (int c = 1; c < chunks; c++)
    if (base[c * slot + head_dim] > max_val)
      max_val = base[c * slot + head_dim];
  float sum = 0.0f;
  memset(o, 0, head_dim * sizeof(float));
  for (int c = 0; c < chunks; c++) {
    const float *chunk = base + c * slot;
    if (partial_empty(chunk, head_dim))
      continue; /* a row with no position in this chunk (tiled, E2); split never writes one, so it is unaffected */
    float weight = expf(chunk[head_dim] - max_val);
    sum += weight * chunk[head_dim + 1];
    for (int i = 0; i < head_dim; i++)
      o[i] += weight * chunk[i];
  }
  for (int i = 0; i < head_dim; i++)
    o[i] /= sum;
}

void k_attention_merge_partials(const float *base, size_t slot, int chunks, int head_dim, float *o) {
  merge_chunks(base, slot, chunks, head_dim, o);
}

/* Opt-in per-task timing for the attention benchmark (E0). Unset (the default), a task costs one atomic load. */
static _Atomic(AttentionTaskHook) attention_task_hook;
static void *attention_task_ctx;

void k_attention_set_task_hook(AttentionTaskHook hook, void *ctx) {
  attention_task_ctx = ctx;
  atomic_store_explicit(&attention_task_hook, hook, memory_order_release);
}

#if HAVE_AVX512
typedef struct {
  float *out;
  const float *q, *key_cache, *value_cache;
  float *att, *partial;
  int n_heads, head_dim, seq_len, positions, kv_mul, kv_dim, chunks, tasks;
  int group; /* 1: the group kernel (E5) runs the (KV group, chunk) tasks instead of the per-head chunk kernel */
  size_t slot, head_stride;
  size_t pos_stride, head_base; /* K/V layout (E5.2): floats between positions of a head, and between KV heads' bases */
  atomic_int next_task; /* team-local queue over (kv group, chunk) tasks */
  Team *team;           /* the barrier's team: workers cannot look it up (thread_team is the coordinator's) */
} SplitKJob;

static void attention_task(const SplitKJob *j, int task) {
  int g = task / j->chunks, c = task % j->chunks;
  int t0 = (int)((long long)j->positions * c / j->chunks), t1 = (int)((long long)j->positions * (c + 1) / j->chunks);
  int h0 = g * j->kv_mul;
  if (j->group) {
    attention_group_avx512(j->q + (size_t)h0 * j->head_dim, j->kv_mul, j->head_dim, t0, t1,
                           j->key_cache + (size_t)g * j->head_base, j->value_cache + (size_t)g * j->head_base,
                           (int)j->pos_stride, j->partial + h0 * j->head_stride + c * j->slot, j->head_stride);
    return;
  }
  attention_chunk_avx512(j->q + (size_t)h0 * j->head_dim, j->key_cache + (size_t)g * j->head_base,
                         j->value_cache + (size_t)g * j->head_base, (int)j->pos_stride, j->head_dim, j->kv_mul, t0, t1,
                         j->att + (size_t)h0 * j->seq_len + t0, j->seq_len,
                         j->partial + h0 * j->head_stride + c * j->slot, j->head_stride);
}

static void attention_merge(const SplitKJob *j, int h) {
  merge_chunks(j->partial + h * j->head_stride, j->slot, j->chunks, j->head_dim, j->out + (size_t)h * j->head_dim);
}

/* The split-K task with the benchmark hook (see batch_run); a decode task is one row. */
static void attention_run(const SplitKJob *j, int task, int worker) {
  AttentionTaskHook hook = atomic_load_explicit(&attention_task_hook, memory_order_acquire);
  if (!hook) {
    attention_task(j, task);
    return;
  }
  int c = task % j->chunks, t0 = (int)((long long)j->positions * c / j->chunks),
      t1 = (int)((long long)j->positions * (c + 1) / j->chunks);
  AttentionTaskInfo info = {.worker = worker, .task = task, .rows = 1, .span = t1 - t0,
                            .positions = t1 - t0};
  info.begin = monotonic_seconds();
  attention_task(j, task);
  info.end = monotonic_seconds();
  hook(attention_task_ctx, &info);
}

static void attention_team(const void *p, int worker, int workers) {
  SplitKJob *j = (SplitKJob *)p;
  for (int task; (task = atomic_fetch_add_explicit(&j->next_task, 1, memory_order_relaxed)) < j->tasks;)
    attention_run(j, task, worker);
  team_barrier(j->team); /* every chunk of every head is in `partial` before any head is merged */
  for (int h = (int)((long long)j->n_heads * worker / workers);
       h < (int)((long long)j->n_heads * (worker + 1) / workers); h++)
    attention_merge(j, h);
}
#endif

/* The split-K decode path; `group` selects the per-group kernel (E5) for the (KV group, chunk) tasks. The chunk plan,
 * the partial slots and the merge are the same for both. */
static void attention_decode_splitk(float *out, const float *q, const float *key_cache, const float *value_cache,
                                    float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                    int seq_len, int position, int group, size_t pos_stride, size_t head_base) {
#if HAVE_AVX512
  if (kernels_isa == KERNELS_AVX512 && head_dim <= 256) {
    int positions = position + 1, kv_mul = n_heads / n_kv_heads, kv_dim = n_kv_heads * head_dim;
    int chunks = attention_chunks(positions, n_kv_heads, work->max_chunks, work->workers);
    SplitKJob job = {.out = out,
                     .q = q,
                     .key_cache = key_cache,
                     .value_cache = value_cache,
                     .att = att,
                     .partial = work->partial,
                     .n_heads = n_heads,
                     .head_dim = head_dim,
                     .seq_len = seq_len,
                     .positions = positions,
                     .kv_mul = kv_mul,
                     .kv_dim = kv_dim,
                     .pos_stride = pos_stride,
                     .head_base = head_base,
                     .chunks = chunks,
                     .tasks = n_kv_heads * chunks,
                     .group = group,
                     .slot = (size_t)head_dim + 2,
                     .head_stride = (size_t)work->max_chunks * ((size_t)head_dim + 2)};
    atomic_init(&job.next_task, 0);
    job.team = current_team;
    if (current_team) {
      team_run(current_team, attention_team, &job);
      return;
    }
    for (int task = 0; task < job.tasks; task++) /* no team (unit tests): serial */
      attention_run(&job, task, 0);
    for (int h = 0; h < n_heads; h++)
      attention_merge(&job, h);
    return;
  }
#endif
  (void)work, (void)group;
  attention_scalar(out, q, key_cache, value_cache, att, n_heads, n_kv_heads, head_dim, seq_len, position, pos_stride,
                   head_base);
}

static void attention_decode_split(float *out, const float *q, const float *key_cache, const float *value_cache,
                                   float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                   int seq_len, int position, size_t pos_stride, size_t head_base) {
  attention_decode_splitk(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, position,
                          0, pos_stride, head_base);
}

/* LLAMA_ATTN_DECODE=group (E5). One note per process when the call goes to `split`: same precision, different speed. */
static void attention_decode_group(float *out, const float *q, const float *key_cache, const float *value_cache,
                                   float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                   int seq_len, int position, size_t pos_stride, size_t head_base) {
  const char *why = NULL;
#if HAVE_AVX512
  if (kernels_isa != KERNELS_AVX512)
    why = "the AVX-512 kernels are not active";
  else if (head_dim > GROUP_MAX_HD || n_heads / n_kv_heads > GROUP_MAX_Q)
    why = "head_dim above 256 or more than 16 query heads per KV head is not supported";
#else
  why = "the AVX-512 kernels are not built";
#endif
  if (why) {
    static atomic_flag noted = ATOMIC_FLAG_INIT;
    if (!atomic_flag_test_and_set(&noted))
      LOG(LOG_INFO, "LLAMA_ATTN_DECODE=group: %s, running split attention instead", why);
  }
  attention_decode_splitk(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, position,
                          why == NULL, pos_stride, head_base);
}

// ----------------------------------------------------------------------------
// Batched causal attention (C3)

typedef void (*ChunkFn)(const float *q, const float *kc, const float *vc, int kv_dim, int head_dim, int heads, int t0,
                        int t1, float *scores, size_t score_stride, float *partial, size_t head_stride);

/* Scalar reference of the chunk kernel: one GQA group over positions [t0, t1), writing per query head the
 * unnormalized value sum, the local maximum and the local sum of exponentials, like attention_chunk_avx512. */
static void attention_chunk_portable(const float *q, const float *kc, const float *vc, int kv_dim, int head_dim,
                                     int heads, int t0, int t1, float *scores, size_t score_stride, float *partial,
                                     size_t head_stride) {
  float divisor = sqrtf((float)head_dim);
  for (int h = 0; h < heads; h++) {
    const float *qh = q + (size_t)h * head_dim;
    float *a = scores + (size_t)h * score_stride, max_val = -FLT_MAX;
    for (int t = t0; t < t1; t++) {
      const float *k = kc + (size_t)t * kv_dim;
      float score = 0.0f;
      for (int i = 0; i < head_dim; i++)
        score += qh[i] * k[i];
      a[t - t0] = score / divisor;
      if (a[t - t0] > max_val)
        max_val = a[t - t0];
    }
    float *o = partial + (size_t)h * head_stride, sum = 0.0f;
    memset(o, 0, head_dim * sizeof(float));
    for (int t = t0; t < t1; t++) {
      float e = expf(a[t - t0] - max_val);
      sum += e;
      const float *v = vc + (size_t)t * kv_dim;
      for (int i = 0; i < head_dim; i++)
        o[i] += e * v[i];
    }
    o[head_dim] = max_val;
    o[head_dim + 1] = sum;
  }
}

enum { BATCH_MAX_GROUP = 16, BATCH_MAX_CHUNK = 256 };

typedef struct {
  float *out;
  const float *q, *key_cache, *value_cache;
  float *partial;
  ChunkFn chunk;
  int n_heads, n_kv_heads, head_dim, kv_mul, pos_stride, dim, limit, workers;
  int pos0, row0, rows; /* this wave: rows row0..row0+rows-1 of the microbatch */
  int chunks_wave, blocks, tasks, row_block;
  size_t slot, head_stride, row_stride, head_base; /* K/V layout: floats between positions / between KV heads' bases */
  atomic_int next_task; /* team-local queue over (kv group, chunk, row block) tasks */
  Team *team;           /* the barrier's team: workers cannot look it up (thread_team is the coordinator's) */
} BatchJob;

static int batch_row_chunks(const BatchJob *j, int row) {
  return attention_chunks(j->pos0 + row + 1, j->n_kv_heads, j->limit, j->workers);
}

static void batch_split(const BatchJob *j, int task, int *g, int *c, int *first, int *last) {
  int block = task % j->blocks, rest = task / j->blocks;
  *c = rest % j->chunks_wave, *g = rest / j->chunks_wave;
  *first = block * j->row_block, *last = *first + j->row_block < j->rows ? *first + j->row_block : j->rows;
}

/* One (kv group, chunk, block of rows) task. Rows of a block read the same K/V chunk back to back, so it stays in L2;
 * each row uses exactly the chunk plan k_attention would use at its position, and positions past the row are masked
 * simply by never being part of its chunks. */
static void batch_task(const BatchJob *j, int task) {
  int g, c, first, last;
  batch_split(j, task, &g, &c, &first, &last);
  float scores[BATCH_MAX_GROUP * BATCH_MAX_CHUNK];
  for (int rr = first; rr < last; rr++) {
    int row = j->row0 + rr, positions = j->pos0 + row + 1, chunks = batch_row_chunks(j, row);
    if (c >= chunks)
      continue;
    int t0 = (int)((long long)positions * c / chunks), t1 = (int)((long long)positions * (c + 1) / chunks);
    int h0 = g * j->kv_mul;
    j->chunk(j->q + (size_t)row * j->dim + (size_t)h0 * j->head_dim, j->key_cache + (size_t)g * j->head_base,
             j->value_cache + (size_t)g * j->head_base, j->pos_stride, j->head_dim, j->kv_mul, t0, t1, scores,
             (size_t)(t1 - t0), j->partial + (size_t)rr * j->row_stride + (size_t)h0 * j->head_stride + c * j->slot,
             j->head_stride);
  }
}

/* What a task reads, for the hook: rows that take part, positions scored, and the K/V span the block streams (the rows'
 * chunks overlap, so the block pulls about min(t0)..max(t1) once). */
static void batch_extent(const BatchJob *j, int task, AttentionTaskInfo *info) {
  int g, c, first, last, lo = INT_MAX, hi = 0;
  batch_split(j, task, &g, &c, &first, &last);
  for (int rr = first; rr < last; rr++) {
    int positions = j->pos0 + j->row0 + rr + 1, chunks = batch_row_chunks(j, j->row0 + rr);
    if (c >= chunks)
      continue;
    int t0 = (int)((long long)positions * c / chunks), t1 = (int)((long long)positions * (c + 1) / chunks);
    info->rows++, info->positions += t1 - t0;
    lo = t0 < lo ? t0 : lo, hi = t1 > hi ? t1 : hi;
  }
  info->span = hi > lo ? hi - lo : 0;
}

static void batch_run(const BatchJob *j, int task, int worker) {
  AttentionTaskHook hook = atomic_load_explicit(&attention_task_hook, memory_order_acquire);
  if (!hook) {
    batch_task(j, task);
    return;
  }
  AttentionTaskInfo info = {.worker = worker, .wave = j->row0, .task = task};
  batch_extent(j, task, &info);
  info.begin = monotonic_seconds();
  batch_task(j, task);
  info.end = monotonic_seconds();
  hook(attention_task_ctx, &info);
}

static void batch_merge(const BatchJob *j, int item) {
  int rr = item / j->n_heads, h = item % j->n_heads, row = j->row0 + rr;
  merge_chunks(j->partial + (size_t)rr * j->row_stride + (size_t)h * j->head_stride, j->slot, batch_row_chunks(j, row),
               j->head_dim, j->out + (size_t)row * j->dim + (size_t)h * j->head_dim);
}

static void batch_team(const void *p, int worker, int workers) {
  BatchJob *j = (BatchJob *)p;
  for (int task; (task = atomic_fetch_add_explicit(&j->next_task, 1, memory_order_relaxed)) < j->tasks;)
    batch_run(j, task, worker);
  team_barrier(j->team); /* every chunk of every row and head is in `partial` before any is merged */
  int items = j->rows * j->n_heads;
  for (int item = (int)((long long)items * worker / workers); item < (int)((long long)items * (worker + 1) / workers);
       item++)
    batch_merge(j, item);
}

static void attention_batch_driver(float *out, const float *q, const float *key_cache, const float *value_cache,
                                   AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int pos0, int m,
                                   size_t pos_stride, size_t head_base, ChunkFn chunk) {
  BatchJob job = {.row_block = BATCH_ROW_BLOCK,
                  .out = out,
                  .q = q,
                  .key_cache = key_cache,
                  .value_cache = value_cache,
                  .partial = work->partial,
                  .chunk = chunk,
                  .n_heads = n_heads,
                  .n_kv_heads = n_kv_heads,
                  .head_dim = head_dim,
                  .kv_mul = n_heads / n_kv_heads,
                  .pos_stride = (int)pos_stride,
                  .head_base = head_base,
                  .dim = n_heads * head_dim,
                  .limit = work->max_chunks,
                  .workers = work->workers,
                  .pos0 = pos0,
                  .slot = (size_t)head_dim + 2};
  for (int row0 = 0; row0 < m;) {
    /* A wave holds as many rows of partials as the workspace allows; chunk counts only grow with the position, so the
     * last row of the wave sets the slots per head. */
    int rows = m - row0;
    for (;;) {
      job.row0 = row0, job.rows = rows;
      job.chunks_wave = batch_row_chunks(&job, row0 + rows - 1);
      if (rows == 1 || (size_t)rows * n_heads * job.chunks_wave * job.slot <= work->capacity)
        break;
      rows = (int)(work->capacity / ((size_t)n_heads * job.chunks_wave * job.slot));
      rows = rows < 1 ? 1 : rows;
    }
    job.head_stride = (size_t)job.chunks_wave * job.slot;
    job.row_stride = (size_t)n_heads * job.head_stride;
    job.blocks = (rows + job.row_block - 1) / job.row_block;
    job.tasks = n_kv_heads * job.chunks_wave * job.blocks;
    atomic_init(&job.next_task, 0);
    job.team = current_team;
    if (current_team) {
      team_run(current_team, batch_team, &job);
    } else { /* no team (unit tests): serial */
      for (int task = 0; task < job.tasks; task++)
        batch_run(&job, task, 0);
      for (int item = 0; item < job.rows * n_heads; item++)
        batch_merge(&job, item);
    }
    row0 += rows;
  }
}

static int batch_supported(int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0, int m) {
  (void)seq_len;
  return head_dim <= 256 && n_heads / n_kv_heads <= BATCH_MAX_GROUP && m > 1 && pos0 >= 0;
}

static void attention_batch_split(float *out, const float *q, const float *key_cache, const float *value_cache,
                                  float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                  int seq_len, int pos0, int m, size_t pos_stride, size_t head_base) {
#if HAVE_AVX512
  if (kernels_isa == KERNELS_AVX512 && batch_supported(n_heads, n_kv_heads, head_dim, seq_len, pos0, m)) {
    attention_batch_driver(out, q, key_cache, value_cache, work, n_heads, n_kv_heads, head_dim, pos0, m, pos_stride,
                           head_base, attention_chunk_avx512);
    return;
  }
#endif
  /* The scalar kernels have no cross-row sharing to exploit: one row at a time, exactly the per-row call. */
  size_t dim = (size_t)n_heads * head_dim;
  for (int i = 0; i < m; i++)
    attention_decode_split(out + i * dim, q + i * dim, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim,
                           seq_len, pos0 + i, pos_stride, head_base);
}

void k_attention_batch_portable_layout(float *out, const float *q, const float *key_cache, const float *value_cache,
                                       AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len,
                                       int pos0, int m, size_t pos_stride, size_t head_base) {
  if (!batch_supported(n_heads, n_kv_heads, head_dim, seq_len, pos0, m))
    fail("portable batched attention needs head_dim <= 256, at most %d heads per KV head and m > 1", BATCH_MAX_GROUP);
  if (pos_stride < (size_t)head_dim || pos_stride > INT_MAX)
    fail("k_attention_batch_portable_layout: position stride %zu is not in [head_dim, INT_MAX]", pos_stride);
  attention_batch_driver(out, q, key_cache, value_cache, work, n_heads, n_kv_heads, head_dim, pos0, m, pos_stride,
                         head_base, attention_chunk_portable);
}

void k_attention_batch_portable(float *out, const float *q, const float *key_cache, const float *value_cache,
                                AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0,
                                int m) {
  k_attention_batch_portable_layout(out, q, key_cache, value_cache, work, n_heads, n_kv_heads, head_dim, seq_len, pos0,
                                    m, (size_t)n_kv_heads * head_dim, head_dim);
}

// ----------------------------------------------------------------------------
// Tiled batched attention (E2)

#if HAVE_AVX512
typedef struct {
  int g, block, chunk;
} TiledTask;

typedef struct {
  float *out;
  const float *q, *key_cache, *value_cache;
  float *partial;
  int n_heads, n_kv_heads, head_dim, kv_mul, pos_stride, dim, limit, workers;
  int pos0, m, row0, rows; /* this wave: rows row0..row0+rows-1 of the microbatch of m rows */
  int rblock, chunks_wave, tasks;
  const TiledTask *order; /* the tasks of the wave, in the order they are claimed */
  size_t slot, head_stride, row_stride, head_base; /* K/V layout: floats between positions / between KV heads' bases */
  atomic_int next_task;
  Team *team;
} TiledJob;

/* The chunk plan of a block of rows: chunk length (a multiple of the position tile) and chunk count. It depends only on
 * the block's nominal last row, never on the wave that happens to contain the block, so results do not depend on how a
 * microbatch is cut into waves. */
static void tiled_plan(const TiledJob *j, int block, int *length, int *chunks) {
  int end = (block + 1) * j->rblock < j->m ? (block + 1) * j->rblock : j->m, positions = j->pos0 + end;
  int wanted = attention_chunks(positions, j->n_kv_heads, j->limit, j->workers);
  int len = ((positions + wanted - 1) / wanted + TILED_POS - 1) / TILED_POS * TILED_POS;
  *length = len;
  *chunks = (positions + len - 1) / len;
}

/* What a task covers: the first and last row of its block that belong to the wave, and its chunk [t0, t1). */
static void tiled_extent(const TiledJob *j, const TiledTask *t, int *first, int *last, int *t0, int *t1) {
  int length, chunks;
  tiled_plan(j, t->block, &length, &chunks);
  int block_end = (t->block + 1) * j->rblock < j->m ? (t->block + 1) * j->rblock : j->m;
  *first = t->block * j->rblock > j->row0 ? t->block * j->rblock : j->row0;
  *last = block_end < j->row0 + j->rows ? block_end : j->row0 + j->rows;
  *t0 = t->chunk * length;
  *t1 = (t->chunk + 1) * length < j->pos0 + block_end ? (t->chunk + 1) * length : j->pos0 + block_end;
}

static void tiled_task(const TiledJob *j, const TiledTask *t) {
  int first, last, t0, t1;
  tiled_extent(j, t, &first, &last, &t0, &t1);
  int h0 = t->g * j->kv_mul, rr = first - j->row0;
  attention_tiled_avx512(j->q + (size_t)first * j->dim + (size_t)h0 * j->head_dim, (size_t)j->dim, last - first,
                         j->kv_mul, j->head_dim, j->pos0 + first + 1, t0, t1,
                         j->key_cache + (size_t)t->g * j->head_base, j->value_cache + (size_t)t->g * j->head_base,
                         j->pos_stride,
                         j->partial + (size_t)rr * j->row_stride + (size_t)h0 * j->head_stride + t->chunk * j->slot,
                         j->row_stride, j->head_stride);
}

static void tiled_run(const TiledJob *j, int index, int worker) {
  const TiledTask *t = &j->order[index];
  AttentionTaskHook hook = atomic_load_explicit(&attention_task_hook, memory_order_acquire);
  if (!hook) {
    tiled_task(j, t);
    return;
  }
  AttentionTaskInfo info = {.worker = worker, .wave = j->row0, .task = index};
  int first, last, t0, t1;
  tiled_extent(j, t, &first, &last, &t0, &t1);
  for (int row = first; row < last; row++) {
    int end = t1 < j->pos0 + row + 1 ? t1 : j->pos0 + row + 1;
    if (end > t0)
      info.rows++, info.positions += end - t0;
  }
  info.span = t1 > t0 ? t1 - t0 : 0;
  info.begin = monotonic_seconds();
  tiled_task(j, t);
  info.end = monotonic_seconds();
  hook(attention_task_ctx, &info);
}

static void tiled_merge(const TiledJob *j, int item) {
  int rr = item / j->n_heads, h = item % j->n_heads, row = j->row0 + rr, length, chunks;
  tiled_plan(j, row / j->rblock, &length, &chunks);
  merge_chunks(j->partial + (size_t)rr * j->row_stride + (size_t)h * j->head_stride, j->slot, chunks, j->head_dim,
               j->out + (size_t)row * j->dim + (size_t)h * j->head_dim);
}

static void tiled_team(const void *p, int worker, int workers) {
  TiledJob *j = (TiledJob *)p;
  for (int task; (task = atomic_fetch_add_explicit(&j->next_task, 1, memory_order_relaxed)) < j->tasks;)
    tiled_run(j, task, worker);
  team_barrier(j->team); /* every chunk of every row and head is in `partial` before any is merged */
  int items = j->rows * j->n_heads;
  for (int item = (int)((long long)items * worker / workers); item < (int)((long long)items * (worker + 1) / workers);
       item++)
    tiled_merge(j, item);
}

/* The tasks of a wave, claimed in plan order (E2: faster than longest first at 2048 and 8192 tokens). */
static TiledTask *tiled_tasks(TiledJob *j) {
  int block0 = j->row0 / j->rblock, block1 = (j->row0 + j->rows - 1) / j->rblock, count = 0;
  TiledTask *list = NULL;
  for (int pass = 0; pass < 2; pass++) {
    if (pass) {
      list = malloc((size_t)count * sizeof(*list));
      if (!list)
        fail("out of memory for the attention task list");
      count = 0;
    }
    for (int block = block0; block <= block1; block++) {
      int length, chunks;
      tiled_plan(j, block, &length, &chunks);
      for (int g = 0; g < j->n_kv_heads; g++)
        for (int c = 0; c < chunks; c++) {
          if (pass)
            list[count] = (TiledTask){.g = g, .block = block, .chunk = c};
          count++;
        }
    }
  }
  j->tasks = count;
  return list;
}

static void attention_tiled_driver(float *out, const float *q, const float *key_cache, const float *value_cache,
                                   AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int pos0, int m,
                                   size_t pos_stride, size_t head_base) {
  int kv_mul = n_heads / n_kv_heads, queries = TILED_QUERIES / kv_mul;
  TiledJob job = {.out = out,
                  .q = q,
                  .key_cache = key_cache,
                  .value_cache = value_cache,
                  .partial = work->partial,
                  .n_heads = n_heads,
                  .n_kv_heads = n_kv_heads,
                  .head_dim = head_dim,
                  .kv_mul = kv_mul,
                  .pos_stride = (int)pos_stride,
                  .head_base = head_base,
                  .dim = n_heads * head_dim,
                  .limit = work->max_chunks,
                  .workers = work->workers,
                  .pos0 = pos0,
                  .m = m,
                  .rblock = queries < 1 ? 1 : queries,
                  .slot = (size_t)head_dim + 2};
  for (int row0 = 0; row0 < m;) {
    /* A wave holds as many rows of partials as the workspace allows. The slots per head follow the plan of the last
     * block of the wave; chunk counts only grow with the position. */
    int rows = m - row0;
    for (;;) {
      job.row0 = row0, job.rows = rows;
      int last_end = ((row0 + rows - 1) / job.rblock + 1) * job.rblock;
      job.chunks_wave = attention_chunks(pos0 + (last_end < m ? last_end : m), n_kv_heads, job.limit, job.workers);
      if (rows == 1 || (size_t)rows * n_heads * job.chunks_wave * job.slot <= work->capacity)
        break;
      rows = (int)(work->capacity / ((size_t)n_heads * job.chunks_wave * job.slot));
      rows = rows < 1 ? 1 : rows;
    }
    job.head_stride = (size_t)job.chunks_wave * job.slot;
    job.row_stride = (size_t)n_heads * job.head_stride;
    TiledTask *order = tiled_tasks(&job);
    job.order = order;
    atomic_init(&job.next_task, 0);
    job.team = current_team;
    if (current_team) {
      team_run(current_team, tiled_team, &job);
    } else { /* no team (unit tests): serial */
      for (int task = 0; task < job.tasks; task++)
        tiled_run(&job, task, 0);
      for (int item = 0; item < job.rows * n_heads; item++)
        tiled_merge(&job, item);
    }
    free(order);
    row0 += rows;
  }
}
#endif

/* Set when the batched variant was chosen explicitly (LLAMA_ATTN or kernels_attention_select), not by default. */
static int attention_batch_requested;

/* One note per process when `tiled` hands a call to `split`: the precision is the same, only the speed differs. It is
 * an INFO note when tiled was asked for and a DEBUG note when it is only the default (every host without AVX-512). */
static void attention_tiled_fallback(const char *reason) {
  static atomic_flag noted = ATOMIC_FLAG_INIT;
  if (!atomic_flag_test_and_set(&noted))
    LOG(attention_batch_requested ? LOG_INFO : LOG_DEBUG, "LLAMA_ATTN=tiled: %s, running split attention instead",
        reason);
}

static void attention_batch_tiled(float *out, const float *q, const float *key_cache, const float *value_cache,
                                  float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                  int seq_len, int pos0, int m, size_t pos_stride, size_t head_base) {
  if (m > 1) { /* a single row is the decode path, whatever the variant */
#if HAVE_AVX512
    if (kernels_isa != KERNELS_AVX512)
      attention_tiled_fallback("the AVX-512 kernels are not active");
    else if (head_dim > 256 || n_heads / n_kv_heads > TILED_MAX_Q)
      attention_tiled_fallback("head_dim above 256 or more than 16 query heads per KV head is not supported");
    else if (batch_supported(n_heads, n_kv_heads, head_dim, seq_len, pos0, m)) {
      attention_tiled_driver(out, q, key_cache, value_cache, work, n_heads, n_kv_heads, head_dim, pos0, m, pos_stride,
                             head_base);
      return;
    }
#else
    attention_tiled_fallback("the AVX-512 kernels are not built");
#endif
  }
  attention_batch_split(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, pos0, m,
                        pos_stride, head_base);
}

// ----------------------------------------------------------------------------
// Attention variant dispatch (E1)

/* Each public attention entry point forwards to one entry of a table, chosen once by kernels_init from LLAMA_ATTN
 * (batched, prefill) and LLAMA_ATTN_DECODE (single row). A later variant is one new table entry plus its function;
 * engine.c and tp.c need no change. The first entry is the default: tiled for the
 * batched path since E4 (owner decision, 2026-11-19; split stays selectable with LLAMA_ATTN=split), split for decode. */
typedef void (*AttentionBatchFn)(float *out, const float *q, const float *key_cache, const float *value_cache,
                                 float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                 int seq_len, int pos0, int m, size_t pos_stride, size_t head_base);
typedef void (*AttentionDecodeFn)(float *out, const float *q, const float *key_cache, const float *value_cache,
                                  float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                  int seq_len, int position, size_t pos_stride, size_t head_base);
typedef struct {
  const char *name;
  AttentionBatchFn run;
} AttentionBatchEntry;
typedef struct {
  const char *name;
  AttentionDecodeFn run;
} AttentionDecodeEntry;

static const AttentionBatchEntry attention_batch_table[] = {
    {"tiled", attention_batch_tiled},
    {"split", attention_batch_split},
};
static const AttentionDecodeEntry attention_decode_table[] = {
    {"split", attention_decode_split},
    {"group", attention_decode_group},
};
static const AttentionBatchEntry *attention_batch_active = &attention_batch_table[0];
static const AttentionDecodeEntry *attention_decode_active = &attention_decode_table[0];

int kernels_attention_select(const char *batch, const char *decode) {
  const AttentionBatchEntry *b = NULL;
  const AttentionDecodeEntry *d = NULL;
  for (size_t i = 0; batch && i < sizeof(attention_batch_table) / sizeof(*attention_batch_table); i++)
    if (!strcmp(batch, attention_batch_table[i].name))
      b = &attention_batch_table[i];
  for (size_t i = 0; decode && i < sizeof(attention_decode_table) / sizeof(*attention_decode_table); i++)
    if (!strcmp(decode, attention_decode_table[i].name))
      d = &attention_decode_table[i];
  if ((batch && !b) || (decode && !d))
    return 0;
  if (b) {
    attention_batch_active = b;
    attention_batch_requested = 1;
  }
  if (d)
    attention_decode_active = d;
  return 1;
}

const char *kernels_attention_name(void) {
  return attention_batch_active->name;
}

const char *kernels_attention_decode_name(void) {
  return attention_decode_active->name;
}

/* LLAMA_ATTN / LLAMA_ATTN_DECODE, read once by kernels_init. Unset, empty or "auto" keep the default. */
static void attention_init_from_env(void) {
  const char *batch = getenv("LLAMA_ATTN"), *decode = getenv("LLAMA_ATTN_DECODE");
  attention_batch_active = &attention_batch_table[0];
  attention_decode_active = &attention_decode_table[0];
  attention_batch_requested = 0;
  if (batch && *batch && strcmp(batch, "auto")) {
    const AttentionBatchEntry *found = NULL;
    for (size_t i = 0; i < sizeof(attention_batch_table) / sizeof(*attention_batch_table); i++)
      if (!strcmp(batch, attention_batch_table[i].name))
        found = &attention_batch_table[i];
    if (!found)
      fail("LLAMA_ATTN must be tiled or split");
    attention_batch_active = found;
    attention_batch_requested = 1;
  }
  if (decode && *decode && strcmp(decode, "auto")) {
    const AttentionDecodeEntry *found = NULL;
    for (size_t i = 0; i < sizeof(attention_decode_table) / sizeof(*attention_decode_table); i++)
      if (!strcmp(decode, attention_decode_table[i].name))
        found = &attention_decode_table[i];
    if (!found)
      fail("LLAMA_ATTN_DECODE must be split or group");
    attention_decode_active = found;
  }
}

void k_attention(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                 AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int position) {
  attention_decode_active->run(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len,
                               position, (size_t)n_kv_heads * head_dim, head_dim);
}

void k_attention_layout(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                        AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int position,
                        size_t pos_stride, size_t head_base) {
  if (pos_stride < (size_t)head_dim || pos_stride > INT_MAX)
    fail("k_attention_layout: position stride %zu is not in [head_dim, INT_MAX]", pos_stride);
  attention_decode_active->run(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len,
                               position, pos_stride, head_base);
}

void k_attention_batch(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                       AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0, int m) {
  attention_batch_active->run(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, pos0,
                              m, (size_t)n_kv_heads * head_dim, head_dim);
}

void k_attention_batch_layout(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                              AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0,
                              int m, size_t pos_stride, size_t head_base) {
  if (pos_stride < (size_t)head_dim || pos_stride > INT_MAX)
    fail("k_attention_batch_layout: position stride %zu is not in [head_dim, INT_MAX]", pos_stride);
  attention_batch_active->run(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, pos0,
                              m, pos_stride, head_base);
}

// ----------------------------------------------------------------------------
// K/V cache layout (E5.4): LLAMA_KV_LAYOUT=rows|head

/* head (the default since E5.4): a layer's cache is [kv_head][position][head_dim], each head with room for seq_len
 * positions. rows (the layout before E5.4): [position][kv_head][head_dim]. The cache is the same size and the cache
 * pointer of a layer is the same in both; the results are bitwise equal. */
static int kv_layout_head = 1;

int kernels_kv_layout_head(void) {
  return kv_layout_head;
}

const char *kernels_kv_layout_name(void) {
  return kv_layout_head ? "head" : "rows";
}

int kernels_kv_layout_select(const char *name) {
  if (!strcmp(name, "rows"))
    kv_layout_head = 0;
  else if (!strcmp(name, "head"))
    kv_layout_head = 1;
  else
    return 0;
  return 1;
}

static void kv_layout_init_from_env(void) {
  const char *layout = getenv("LLAMA_KV_LAYOUT");
  kv_layout_head = 1; /* unset, empty and auto are head */
  if (layout && *layout && strcmp(layout, "auto") && !kernels_kv_layout_select(layout))
    fail("LLAMA_KV_LAYOUT must be rows or head");
}

void k_kv_strides(int n_kv_heads, int head_dim, int seq_len, size_t *pos_stride, size_t *head_base) {
  *pos_stride = kv_layout_head ? (size_t)head_dim : (size_t)n_kv_heads * head_dim;
  *head_base = kv_layout_head ? (size_t)seq_len * head_dim : (size_t)head_dim;
}

void k_kv_store(float *cache, const float *src, size_t src_ld, int pos0, int count, int n_kv_heads, int head_dim,
                int seq_len) {
  size_t kv_dim = (size_t)n_kv_heads * head_dim;
  if (!kv_layout_head) {
    for (int r = 0; r < count; r++)
      memcpy(cache + (size_t)(pos0 + r) * kv_dim, src + (size_t)r * src_ld, kv_dim * sizeof(float));
    return;
  }
  for (int r = 0; r < count; r++)
    for (int g = 0; g < n_kv_heads; g++)
      memcpy(cache + (size_t)g * seq_len * head_dim + (size_t)(pos0 + r) * head_dim,
             src + (size_t)r * src_ld + (size_t)g * head_dim, (size_t)head_dim * sizeof(float));
}

void k_attention_kv(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                    AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int position) {
  if (!kv_layout_head) {
    k_attention(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, position);
    return;
  }
  k_attention_layout(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, position,
                     (size_t)head_dim, (size_t)seq_len * head_dim);
}

void k_attention_batch_kv(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                          AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0,
                          int m) {
  if (!kv_layout_head) {
    k_attention_batch(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, pos0, m);
    return;
  }
  k_attention_batch_layout(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, pos0, m,
                           (size_t)head_dim, (size_t)seq_len * head_dim);
}

void kernels_attention_init_env(void) {
  attention_init_from_env();
  kv_layout_init_from_env();
}
