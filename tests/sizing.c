/* Microbatch sizing for the two-socket batched prefill (D1 §4.2): the pure functions of tp.h on the shapes of the study's
 * 8B and 1B models. No files, no threads. */
#include "tp.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static Config llama8b(void) {
  return (Config){.dim = 4096, .hidden_dim = 14336, .n_layers = 32, .n_heads = 32, .n_kv_heads = 8,
                  .vocab_size = 128256, .seq_len = 4096, .head_dim = 128};
}

static Config llama1b(void) {
  return (Config){.dim = 2048, .hidden_dim = 8192, .n_layers = 16, .n_heads = 32, .n_kv_heads = 8,
                  .vocab_size = 128256, .seq_len = 4096, .head_dim = 64};
}

/* The D1 table: payload per exchange is M x 4096 x 4 B, and what a socket reads from its peer per token does not depend on M. */
static void payloads(void) {
  Config c = llama8b();
  assert(tp_exchange_payload_bytes(&c, 16) == 256u << 10);
  assert(tp_exchange_payload_bytes(&c, 64) == 1u << 20);
  assert(tp_exchange_payload_bytes(&c, 256) == 4u << 20);
  assert(tp_peer_bytes_per_token(&c) == 1u << 20);
  Config small = llama1b();
  assert(tp_peer_bytes_per_token(&small) == 2u * 16 * 2048 * 4);
  /* The exchange region holds two partials and the per-element flags of one socket: three lines-aligned vectors. */
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

static TpSizingInput input(void) {
  /* Orders of magnitude of B4, B7 and C7 (modeled, not measured here): 50 us per exchange, 20 GB/s across UPI, a layer
   * shard of about 0.75 GB streamed at 500 GB/s, and a compute-bound layer of 60 us per row (570 tok/s over 32 layers
   * and two sockets). */
  return (TpSizingInput){.work_budget = 1u << 30, .ridge = 32, .exchange_latency_us = 50, .exchange_gbps = 20,
                         .layer_us_per_row = 60, .layer_weight_us = 1500, .max_exchange_fraction = 0.10};
}

static void choice(void) {
  Config c = llama8b();
  const int candidates[] = {16, 32, 64, 128, 256};
  TpSizingRow rows[5];
  TpSizingInput in = input();

  int chosen = tp_choose_batch(&c, 0, 0, &in, candidates, 5, rows);
  for (int i = 0; i < 5; i++) {
    assert(rows[i].rows == candidates[i]);
    assert(rows[i].feasible == (rows[i].fits && rows[i].ridge && rows[i].exchange_ok));
  }
  assert(!rows[0].ridge); /* 16 is below the FP32 ridge */
  assert(rows[1].ridge);
  assert(chosen == 256 && rows[4].feasible);

  /* Condition 1: a work budget just below what 256 rows need drops 256 and keeps 128. */
  in.work_budget = rows[4].work_bytes + rows[4].exchange_bytes - 1;
  chosen = tp_choose_batch(&c, 0, 0, &in, candidates, 5, rows);
  assert(!rows[4].fits && !rows[4].feasible && rows[3].fits && chosen == 128);

  /* Condition 2: the BF16 ridge of 50 rows removes 32 and keeps 64. */
  in = input();
  in.ridge = 50;
  chosen = tp_choose_batch(&c, 0, 0, &in, candidates, 5, rows);
  assert(!rows[1].feasible && rows[2].feasible && chosen == 256);
  in.work_budget = rows[2].work_bytes + rows[2].exchange_bytes;
  assert(tp_choose_batch(&c, 0, 0, &in, candidates, 5, rows) == 64);

  /* Condition 3: a slow exchange (a fixed cost that dominates at small M, then a thin link) keeps only the sizes that
   * amortize it. */
  in = input();
  in.exchange_latency_us = 600;
  in.exchange_gbps = 4;
  in.max_exchange_fraction = 0.25;
  chosen = tp_choose_batch(&c, 0, 0, &in, candidates, 5, rows);
  assert(rows[0].fraction > rows[1].fraction && rows[1].fraction > rows[2].fraction); /* the fixed cost amortizes */
  for (int i = 0; i < 5; i++)
    assert(rows[i].exchange_ok == (rows[i].fraction <= 0.25));
  assert(!rows[0].exchange_ok && !rows[1].exchange_ok && !rows[2].exchange_ok && rows[3].exchange_ok);
  assert(chosen == 256);
  in.work_budget = rows[3].work_bytes + rows[3].exchange_bytes; /* and 256 does not fit: 128 is the largest left */
  assert(tp_choose_batch(&c, 0, 0, &in, candidates, 5, rows) == 128);

  /* Nothing feasible: every size too large for the budget. */
  in = input();
  in.work_budget = 1;
  assert(tp_choose_batch(&c, 0, 0, &in, candidates, 5, rows) == 0);

  /* M x dim must fit the exchange's 32-bit element count. */
  const int huge[] = {1 << 20};
  TpSizingRow big;
  in = input();
  in.work_budget = (size_t)1 << 60;
  assert(tp_choose_batch(&c, 0, 0, &in, huge, 1, &big) == 0 && !big.fits);
}

static void fractions(void) {
  Config c = llama8b();
  TpSizingInput in = input();
  /* Below the ridge a layer costs its weight stream, so the fraction is the exchange over (exchange + stream). */
  double ex16 = 2.0 * (50 + (double)tp_exchange_payload_bytes(&c, 16) / (20 * 1e3));
  assert(tp_exchange_fraction(&c, 16, &in) > ex16 / (ex16 + 1500) - 1e-12 &&
         tp_exchange_fraction(&c, 16, &in) < ex16 / (ex16 + 1500) + 1e-12);
  /* Compute-bound: the fraction tends to the per-row exchange cost over the per-row compute cost, independent of M. */
  double a = tp_exchange_fraction(&c, 256, &in), b = tp_exchange_fraction(&c, 1024, &in);
  assert(a > 0 && b > 0 && b < 1 && (a - b < 0.02 && b - a < 0.02));
}

int main(void) {
  payloads();
  work_region();
  choice();
  fractions();
  puts("sizing: ok");
  return 0;
}
