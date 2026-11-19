#ifndef LLAMA_TP_H
#define LLAMA_TP_H

#include "runtime.h"
#include "shards.h"

/* Two-socket tensor-parallel forward pass (B5, B6): one model instance split into two socket-owned shards, each run
 * by its own worker team, with two cross-socket reductions per layer (B4). Selected by LLAMA_SOCKETS=2, which needs
 * LLAMA_BACKEND=teams and a topology with two sockets (real, or LLAMA_TOPOLOGY=virtual:2xT where the teams are
 * unpinned and the regions portable). Default is 1: nothing in this module runs.
 *
 * The caller is the coordinator of socket 0 (worker 0 of team 0); a thread of this module is the coordinator of
 * socket 1 (worker 0 of team 1). Each coordinator drives its team with the ordinary kernels (kernels_set_thread_team),
 * the two meet in exchange_reduce_add after Wo and W2, and the classifier writes each socket's vocabulary range
 * into one logits buffer. Both residual copies stay bitwise identical (the operand order of the reduction is fixed);
 * LLAMA_TP_CHECK=1 compares them after every layer. */
typedef struct Tp Tp;

/* Packed copies of the seven per-layer matrices of every shard for the AMX tiles (D1), built at load into a ROLE_PACKED
 * region on the shard's own node: BF16 (LLAMA_PRECISION=bf16, FP32 checkpoints) or int8 (AMX-INT8, W8A8). The shards
 * stay: decode reads them. */
typedef enum { TP_PACK_NONE, TP_PACK_BF16, TP_PACK_INT8 } TpPack;

/* LLAMA_SOCKETS validated: 1 when unset; 2 requires LLAMA_BACKEND=teams. */
int tp_sockets_requested(void);
/* The sockets of the topology the pool must be built for (kernels_backend_init), or fails with the reason. */
int tp_check_topology(const char *placement);

/* Plans the shards, reserves and loads the regions, and starts the socket-1 coordinator. The pool must hold two
 * teams. `quantized` selects the Q8_0 path with group size gs; fp32_activations is W8A32 (LLAMA_Q8_ACTIVATIONS=fp32).
 */
Tp *tp_create(const Config *p, const TensorManifest *m, int fd, int quantized, int gs, int fp32_activations,
              int max_batch, TpPack pack);
float *tp_forward(Tp *tp, int token, int pos);
/* D1: `count` consecutive positions from pos0 in one pass over the weights (FP32 and W8A8; W8A32 stays per token,
 * so its capacity is 1).
 * Same return contract as engine_forward_batch: LAST one row, ALL count rows, NONE NULL; the storage belongs to tp. */
float *tp_forward_batch(Tp *tp, const int *tokens, int count, int pos0, LogitsWanted want);

/* Accounting of both sockets' KV caches; with `verify`, also a page-placement query of every cache (A5's evidence). */
void tp_kv_stats(Tp *tp, int position, int verify, KvStats *out);
/* Region sizes, exposed for tests/sizing.c. Per-socket ROLE_WORK bytes for `rows` activation rows (what tp_create
 * reserves), and the bytes of one socket's exchange region for `rows` rows. */
size_t tp_work_bytes(const Config *p, int quantized, int gs, int rows);
size_t tp_exchange_bytes(const Config *p, int rows);

void tp_free(Tp *tp);

#endif
