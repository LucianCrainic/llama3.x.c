/* A software model of the AMX tile registers and the four tile instructions the GEMM microkernels use (ldtilecfg,
 * tilezero, tileloadd/tilestored, tdpbf16ps, tdpbssd, tilerelease), so the exact tile loops, VNNI indexing and
 * accumulation order of src/gemm.c run on any host. Test builds only: gemm.c includes this when GEMM_EMULATE_AMX is
 * defined. The model enforces what the hardware enforces (a loaded configuration, tile shapes that match the
 * instruction's operand rules) so misuse fails loudly instead of computing something plausible.
 *
 * Arithmetic: tdpbssd is exact int32 arithmetic. tdpbf16ps adds the even product, then the odd product, of every K
 * pair to the FP32 accumulator, one rounding per addition and in ascending K order (the order Intel's pseudocode
 * gives); the hardware may differ in the last bit, which the 2e-6 budget of the tests covers. */
#ifndef AMX_EMUL_H
#define AMX_EMUL_H
#include <assert.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  uint8_t palette_id, start_row, reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} EmulTileConfig;

typedef struct {
  int configured;
  uint16_t colsb[8];
  uint8_t rows[8];
  _Alignas(64) uint8_t data[8][16][64];
} EmulTiles;

static _Thread_local EmulTiles emul_tiles;

static inline void emul_tile_loadconfig(const void *config) {
  EmulTileConfig cfg;
  memcpy(&cfg, config, sizeof(cfg));
  assert(cfg.palette_id == 1 && "palette 1 is the only one defined");
  memset(emul_tiles.data, 0, sizeof(emul_tiles.data));
  for (int t = 0; t < 8; t++) {
    assert(cfg.rows[t] <= 16 && cfg.colsb[t] <= 64);
    emul_tiles.rows[t] = cfg.rows[t];
    emul_tiles.colsb[t] = cfg.colsb[t];
  }
  emul_tiles.configured = 1;
}
static inline void emul_tile_release(void) {
  memset(&emul_tiles, 0, sizeof(emul_tiles));
}
static inline int emul_tile_configured(void) {
  return emul_tiles.configured;
}
static inline void emul_tile_zero(int t) {
  assert(emul_tiles.configured);
  memset(emul_tiles.data[t], 0, sizeof(emul_tiles.data[t]));
}
static inline void emul_tile_loadd(int t, const void *base, long stride) {
  assert(emul_tiles.configured);
  for (int r = 0; r < emul_tiles.rows[t]; r++)
    memcpy(emul_tiles.data[t][r], (const char *)base + (long)r * stride, emul_tiles.colsb[t]);
}
static inline void emul_tile_stored(int t, void *base, long stride) {
  assert(emul_tiles.configured);
  for (int r = 0; r < emul_tiles.rows[t]; r++)
    memcpy((char *)base + (long)r * stride, emul_tiles.data[t][r], emul_tiles.colsb[t]);
}

static inline float emul_bf16_to_f32(uint16_t v) {
  uint32_t bits = (uint32_t)v << 16;
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

/* c (rows M, colsb 4N) += a (rows M, colsb 4K) x b (rows K, colsb 4N), where a holds K pairs of BF16 per row and b
 * holds N pairs per row. */
static inline void emul_tile_dpbf16ps(int c, int a, int b) {
  assert(emul_tiles.configured && c != a && c != b && a != b);
  int m = emul_tiles.rows[c], n = emul_tiles.colsb[c] / 4, k = emul_tiles.rows[b];
  assert(emul_tiles.rows[a] == m && emul_tiles.colsb[a] == 4 * k && emul_tiles.colsb[b] == 4 * n);
  for (int i = 0; i < m; i++) {
    float acc[16];
    memcpy(acc, emul_tiles.data[c][i], sizeof(float) * (size_t)n);
    const uint16_t *ar = (const uint16_t *)emul_tiles.data[a][i];
    for (int p = 0; p < k; p++) {
      const uint16_t *br = (const uint16_t *)emul_tiles.data[b][p];
      for (int j = 0; j < n; j++) {
        float even = emul_bf16_to_f32(ar[2 * p]) * emul_bf16_to_f32(br[2 * j]);
        acc[j] += even;
        float odd = emul_bf16_to_f32(ar[2 * p + 1]) * emul_bf16_to_f32(br[2 * j + 1]);
        acc[j] += odd;
      }
    }
    memcpy(emul_tiles.data[c][i], acc, sizeof(float) * (size_t)n);
  }
}

/* c (int32, M x N) += a (int8, M rows of 4K) x b (int8, K rows of 4N), both operands signed. */
static inline void emul_tile_dpbssd(int c, int a, int b) {
  assert(emul_tiles.configured && c != a && c != b && a != b);
  int m = emul_tiles.rows[c], n = emul_tiles.colsb[c] / 4, k = emul_tiles.rows[b];
  assert(emul_tiles.rows[a] == m && emul_tiles.colsb[a] == 4 * k && emul_tiles.colsb[b] == 4 * n);
  for (int i = 0; i < m; i++) {
    int32_t acc[16];
    memcpy(acc, emul_tiles.data[c][i], sizeof(int32_t) * (size_t)n);
    const int8_t *ar = (const int8_t *)emul_tiles.data[a][i];
    for (int p = 0; p < k; p++) {
      const int8_t *br = (const int8_t *)emul_tiles.data[b][p];
      for (int j = 0; j < n; j++)
        for (int q = 0; q < 4; q++)
          acc[j] += (int32_t)ar[4 * p + q] * (int32_t)br[4 * j + q];
    }
    memcpy(emul_tiles.data[c][i], acc, sizeof(int32_t) * (size_t)n);
  }
}
#endif
