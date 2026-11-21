#ifndef LLAMA_TP_H
#define LLAMA_TP_H

#include "loader.h"
#include "runtime.h"
#include "shards.h"

/* The shard engine: the one forward pass of both executables (F3). One socket runs it with S = 1, where every shard is
 * a whole tensor viewed in the loader's weight store and Wo/W2 add straight into the residual (tp_create_local). Two
 * sockets run it tensor-parallel (B5, B6): one model instance split into two socket-owned shards, each run
 * by its own worker team, with two cross-socket reductions per layer (B4). Selected by LLAMA_SOCKETS=2, or by default
 * on a dual-socket HBM node (tp_sockets_requested); it runs on worker teams and a topology with two sockets (real, or
 * LLAMA_TOPOLOGY=virtual:2xT where the teams are unpinned and the regions portable).
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

/* LLAMA_SOCKETS validated (1 or 2). Unset, it is 2 on a discovered topology whose two sockets each have an allowed HBM
 * node, unless LLAMA_TOPOLOGY or LLAMA_PLACEMENT=file|ddr is set; else 1. Resolved once per process, at the first call (kernels_backend_init, before any worker is pinned). */
int tp_sockets_requested(void);
/* Both sockets of the discovered or explicit topology have an HBM node in this job's allowed memory set. */
int tp_hbm_pair(void);
/* The sockets of the topology the pool must be built for (kernels_backend_init), or fails with the reason. */
int tp_check_topology(const char *placement);

/* Plans the shards, reserves and loads the regions, and starts the socket-1 coordinator. The pool must hold two
 * teams. `quantized` selects the Q8_0 path with group size gs; fp32_activations is W8A32 (LLAMA_Q8_ACTIVATIONS=fp32).
 */
Tp *tp_create_planned(const Model *model, const TensorManifest *m, int fd, int quantized, int gs, int fp32_activations,
                      TpPack pack, int plan_only);
void tp_placement_report(FILE *f);
float *tp_all_logits(Tp *tp, int rows);
Tp *tp_create(const Config *p, const TensorManifest *m, int fd, int quantized, int gs, int fp32_activations,
              int max_batch, TpPack pack);
/* One socket: the shards are views into `store` (which must outlive the engine), the KV caches and work buffers live in
 * the loader's run memory and the packed copies in its packed store, so LLAMA_PLACEMENT=file|ddr|hbm places them as
 * before (A4, A5, C5). Runs on the calling thread and the default worker team. */
Tp *tp_create_local(const Config *p, const TensorManifest *m, const WeightStore *store, int quantized, int gs,
                    int fp32_activations, int max_batch, TpPack pack);
/* One token at `pos`. With `logits` 0 the final norm and classifier are skipped (a prompt row whose logits nobody asked
 * for) and NULL is returned. */
float *tp_forward(Tp *tp, int token, int pos, int logits);
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
