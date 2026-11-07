/* A scalar emulation of the handful of AVX-512 intrinsics the GEMM microkernel uses, so its exact code (macros,
 * masks, tails) can run on any host. Test builds only: gemm.c includes this when GEMM_EMULATE_AVX512 is defined. */
#ifndef AVX512_EMUL_H
#define AVX512_EMUL_H
#include <math.h>
#include <stdint.h>
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
#endif
