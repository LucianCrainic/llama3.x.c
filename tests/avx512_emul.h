/* A scalar emulation of the handful of AVX-512 intrinsics the GEMM microkernel uses, so its exact code (macros,
 * masks, tails) can run on any host. Test builds only: gemm.c includes this when GEMM_EMULATE_AVX512 is defined. */
#ifndef AVX512_EMUL_H
#define AVX512_EMUL_H
#include <math.h>
#include <stdint.h>
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
#endif
