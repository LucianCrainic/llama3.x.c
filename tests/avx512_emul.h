/* A scalar emulation of the handful of AVX-512 intrinsics the GEMM microkernel uses, so its exact code (macros,
 * masks, tails) can run on any host. Test builds only: gemm.c includes this when GEMM_EMULATE_AVX512 is defined. */
#ifndef AVX512_EMUL_H
#define AVX512_EMUL_H
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
  float v[16];
} __m512;
typedef uint16_t __mmask16;
static inline __m512 _mm512_setzero_ps(void) {
  return (__m512){{0}};
}
static inline __m512 _mm512_loadu_ps(const float *p) {
  __m512 r;
  for (int i = 0; i < 16; i++)
    r.v[i] = p[i];
  return r;
}
/* Lanes outside the mask are zero and, like the hardware, are never read. */
static inline __m512 _mm512_maskz_loadu_ps(__mmask16 m, const float *p) {
  __m512 r = _mm512_setzero_ps();
  for (int i = 0; i < 16; i++)
    r.v[i] = (m >> i & 1) ? p[i] : 0.0f;
  return r;
}
static inline __m512 _mm512_fmadd_ps(__m512 a, __m512 b, __m512 c) {
  for (int i = 0; i < 16; i++)
    c.v[i] = fmaf(a.v[i], b.v[i], c.v[i]);
  return c;
}
static inline float _mm512_reduce_add_ps(__m512 a) {
  float s = 0;
  for (int i = 0; i < 16; i++)
    s += a.v[i];
  return s;
}
/* The integer subset the batched Q8 microkernel (C6) uses: byte loads, the absolute-value / sign trick, VNNI dpbusd,
 * conversion and the scalar-broadcast multiplies. */
typedef union {
  int8_t i8[64];
  uint8_t u8[64];
  int32_t i32[16];
} __m512i;
typedef uint64_t __mmask64;
static inline __m512i _mm512_setzero_si512(void) {
  return (__m512i){{0}};
}
static inline __m512i _mm512_loadu_si512(const void *p) {
  __m512i r;
  memcpy(&r, p, sizeof(r));
  return r;
}
static inline __m512i _mm512_abs_epi8(__m512i a) {
  for (int i = 0; i < 64; i++)
    a.i8[i] = (int8_t)(a.i8[i] < 0 ? -a.i8[i] : a.i8[i]); /* -128 stays -128, as in hardware; never used here */
  return a;
}
static inline __mmask64 _mm512_movepi8_mask(__m512i a) {
  __mmask64 m = 0;
  for (int i = 0; i < 64; i++)
    m |= (__mmask64)(a.i8[i] < 0) << i;
  return m;
}
/* Lanes with the mask bit set take a - b, the others keep src. */
static inline __m512i _mm512_mask_sub_epi8(__m512i src, __mmask64 k, __m512i a, __m512i b) {
  for (int i = 0; i < 64; i++)
    if (k >> i & 1)
      src.i8[i] = (int8_t)(a.i8[i] - b.i8[i]);
  return src;
}
/* src += sum of four unsigned-by-signed byte products per 32-bit lane. */
static inline __m512i _mm512_dpbusd_epi32(__m512i src, __m512i a, __m512i b) {
  for (int i = 0; i < 16; i++)
    for (int q = 0; q < 4; q++)
      src.i32[i] += (int32_t)a.u8[4 * i + q] * (int32_t)b.i8[4 * i + q];
  return src;
}
static inline __m512 _mm512_cvtepi32_ps(__m512i a) {
  __m512 r;
  for (int i = 0; i < 16; i++)
    r.v[i] = (float)a.i32[i];
  return r;
}
static inline __m512 _mm512_set1_ps(float x) {
  __m512 r;
  for (int i = 0; i < 16; i++)
    r.v[i] = x;
  return r;
}
static inline __m512 _mm512_mul_ps(__m512 a, __m512 b) {
  for (int i = 0; i < 16; i++)
    a.v[i] *= b.v[i];
  return a;
}

/* ---- The rest of what src/kernels_simd.c uses (KERNELS_EMULATE_AVX512): attention, RMSNorm, RoPE, quantization and the
 * Q8 dot products. Same rule as above: lanes outside a mask are never read, results are the architectural ones. */
typedef union {
  int8_t i8[16];
} __m128i;
typedef union {
  int8_t i8[32];
} __m256i;
#define _MM_HINT_T0 3
#define _mm_prefetch(address, hint) ((void)(address), (void)(hint))
#define _MM_FROUND_TO_NEAREST_INT 0
#define _MM_FROUND_TO_ZERO 3
#define _MM_FROUND_NO_EXC 8
#define _CMP_LE_OQ 18
#define _CMP_GE_OQ 29
#define EMUL_LANES(expr)                                                                                               \
  for (int i = 0; i < 16; i++) {                                                                                       \
    expr;                                                                                                              \
  }
static inline __m512 _mm512_add_ps(__m512 a, __m512 b) {
  EMUL_LANES(a.v[i] += b.v[i])
  return a;
}
static inline __m512 _mm512_sub_ps(__m512 a, __m512 b) {
  EMUL_LANES(a.v[i] -= b.v[i])
  return a;
}
static inline __m512 _mm512_div_ps(__m512 a, __m512 b) {
  EMUL_LANES(a.v[i] /= b.v[i])
  return a;
}
/* min/max return the second operand when either is NaN, as the instructions do. */
static inline __m512 _mm512_max_ps(__m512 a, __m512 b) {
  EMUL_LANES(a.v[i] = a.v[i] > b.v[i] ? a.v[i] : b.v[i])
  return a;
}
static inline __m512 _mm512_min_ps(__m512 a, __m512 b) {
  EMUL_LANES(a.v[i] = a.v[i] < b.v[i] ? a.v[i] : b.v[i])
  return a;
}
static inline __m512 _mm512_fnmadd_ps(__m512 a, __m512 b, __m512 c) {
  EMUL_LANES(c.v[i] = fmaf(-a.v[i], b.v[i], c.v[i]))
  return c;
}
static inline __m512 _mm512_mask_add_ps(__m512 src, __mmask16 k, __m512 a, __m512 b) {
  EMUL_LANES(if (k >> i & 1) src.v[i] = a.v[i] + b.v[i])
  return src;
}
static inline __m512 _mm512_mask_max_ps(__m512 src, __mmask16 k, __m512 a, __m512 b) {
  EMUL_LANES(if (k >> i & 1) src.v[i] = a.v[i] > b.v[i] ? a.v[i] : b.v[i])
  return src;
}
static inline float _mm512_reduce_max_ps(__m512 a) {
  float m = a.v[0];
  for (int i = 1; i < 16; i++)
    m = a.v[i] > m ? a.v[i] : m;
  return m;
}
static inline void _mm512_mask_storeu_ps(float *p, __mmask16 k, __m512 a) {
  EMUL_LANES(if (k >> i & 1) p[i] = a.v[i])
}
static inline __m512i _mm512_set1_epi32(int x) {
  __m512i r;
  for (int i = 0; i < 16; i++)
    r.i32[i] = x;
  return r;
}
static inline __m512 _mm512_castsi512_ps(__m512i a) {
  __m512 r;
  memcpy(&r, &a, sizeof(r));
  return r;
}
static inline __m512 _mm512_and_ps(__m512 a, __m512 b) {
  uint32_t x[16], y[16];
  memcpy(x, &a, sizeof(x)), memcpy(y, &b, sizeof(y));
  EMUL_LANES(x[i] &= y[i])
  memcpy(&a, x, sizeof(x));
  return a;
}
static inline __m512 _mm512_or_ps(__m512 a, __m512 b) {
  uint32_t x[16], y[16];
  memcpy(x, &a, sizeof(x)), memcpy(y, &b, sizeof(y));
  EMUL_LANES(x[i] |= y[i])
  memcpy(&a, x, sizeof(x));
  return a;
}
/* Ordered, quiet predicates: false when either operand is NaN. Only the two the kernels use. */
static inline int emul_cmp(float a, float b, int predicate) {
  return predicate == _CMP_LE_OQ ? a <= b : predicate == _CMP_GE_OQ ? a >= b : (abort(), 0);
}
static inline __mmask16 _mm512_cmp_ps_mask(__m512 a, __m512 b, int predicate) {
  __mmask16 k = 0;
  EMUL_LANES(k |= (__mmask16)(emul_cmp(a.v[i], b.v[i], predicate) << i))
  return k;
}
static inline __mmask16 _mm512_mask_cmp_ps_mask(__mmask16 m, __m512 a, __m512 b, int predicate) {
  return (__mmask16)(_mm512_cmp_ps_mask(a, b, predicate) & m);
}
/* The immediate's low two bits pick the rounding (0 nearest even, 1 down, 2 up, 3 toward zero); no fraction bits. */
static inline __m512 _mm512_roundscale_ps(__m512 a, int imm) {
  EMUL_LANES(a.v[i] = (imm & 3) == 0 ? nearbyintf(a.v[i]) : (imm & 3) == 1 ? floorf(a.v[i]) : (imm & 3) == 2 ? ceilf(a.v[i]) : truncf(a.v[i]))
  return a;
}
/* a * 2^floor(b), exact like vscalefps for the finite range the kernels use. */
static inline __m512 _mm512_scalef_ps(__m512 a, __m512 b) {
  EMUL_LANES(a.v[i] = ldexpf(a.v[i], (int)floorf(b.v[i])))
  return a;
}
/* Within each 128-bit lane, element j takes element (imm >> 2j) & 3. */
static inline __m512 _mm512_permute_ps(__m512 a, int imm) {
  __m512 r;
  EMUL_LANES(r.v[i] = a.v[(i & ~3) + ((imm >> (2 * (i & 3))) & 3)])
  return r;
}
static inline __m512i _mm512_cvttps_epi32(__m512 a) {
  __m512i r;
  EMUL_LANES(r.i32[i] = (int32_t)truncf(a.v[i]))
  return r;
}
static inline __m128i _mm512_cvtepi32_epi8(__m512i a) {
  __m128i r;
  EMUL_LANES(r.i8[i] = (int8_t)a.i32[i])
  return r;
}
static inline __m512i _mm512_cvtepi8_epi32(__m128i a) {
  __m512i r;
  EMUL_LANES(r.i32[i] = a.i8[i])
  return r;
}
static inline __m128i _mm_loadu_si128(const __m128i *p) {
  __m128i r;
  memcpy(&r, p, sizeof(r));
  return r;
}
static inline void _mm_storeu_si128(__m128i *p, __m128i a) {
  memcpy(p, &a, sizeof(a));
}
static inline __m256i _mm256_loadu_si256(const __m256i *p) {
  __m256i r;
  memcpy(&r, p, sizeof(r));
  return r;
}
static inline __m128i _mm256_castsi256_si128(__m256i a) {
  __m128i r;
  memcpy(&r, a.i8, sizeof(r));
  return r;
}
static inline __m128i _mm256_extracti128_si256(__m256i a, int index) {
  __m128i r;
  memcpy(&r, a.i8 + 16 * index, sizeof(r));
  return r;
}
#undef EMUL_LANES
#endif
