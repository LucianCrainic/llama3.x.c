/* Region sizes of the two-socket batched prefill (D1 §4.2): tp_work_bytes and tp_exchange_bytes on the 8B shape. No
 * files, no threads. */
#include "tp.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static Config llama8b(void) {
  return (Config){.dim = 4096, .hidden_dim = 14336, .n_layers = 32, .n_heads = 32, .n_kv_heads = 8,
                  .vocab_size = 128256, .seq_len = 4096, .head_dim = 128};
}

/* The exchange region holds two partials and the per-element flags of one socket: three lines-aligned vectors. */
static void exchange_region(void) {
  Config c = llama8b();
  assert(tp_exchange_bytes(&c, 256) == 3u * 4096 * 256 * sizeof(float));
}

/* The work region is the activations of M rows plus a constant (attention scratch, workspace), so it grows by exactly the
 * per-row activation bytes, and the quantized layout adds the padded int8 rows. */
static void work_region(void) {
  Config c = llama8b();
  size_t per_row = (size_t)(2 * 4096 + 14336 / 2 + 4096 / 2 + 2 * 128) * sizeof(float);
  size_t w1 = tp_work_bytes(&c, 0, 0, 1), w64 = tp_work_bytes(&c, 0, 0, 64), w256 = tp_work_bytes(&c, 0, 0, 256);
  assert(w1 < w64 && w64 < w256);
  /* Activations are the only part that depends on M while the attention workspace is within its 32 MiB budget. */
  assert(w256 - w64 >= 192 * per_row);
  assert(w256 - w64 <= 192 * per_row + (64u << 20));
  size_t q = tp_work_bytes(&c, 1, 128, 64);
  size_t padded_row = 4096 + 4096 / 128 * sizeof(float) + 7168 + 7168 / 128 * sizeof(float);
  assert(q - w64 == 64 * padded_row); /* 64 rows are already whole 32-row blocks */
  assert(tp_work_bytes(&c, 1, 128, 33) - tp_work_bytes(&c, 0, 0, 33) == 64 * padded_row); /* 33 rows pad to 64 */
}

int main(void) {
  exchange_region();
  work_region();
  puts("sizing: ok");
  return 0;
}
