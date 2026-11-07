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

/* LLAMA_SOCKETS validated: 1 when unset; 2 requires LLAMA_BACKEND=teams. */
int tp_sockets_requested(void);
/* The sockets of the topology the pool must be built for (kernels_backend_init), or fails with the reason. */
int tp_check_topology(const char *placement);

/* Plans the shards, reserves and loads the regions, and starts the socket-1 coordinator. The pool must hold two
 * teams. `quantized` selects the Q8_0 path with group size gs. */
Tp *tp_create(const Config *p, const TensorManifest *m, int fd, int quantized, int gs);
float *tp_forward(Tp *tp, int token, int pos);
void tp_kv_stats(const Tp *tp, int position, KvStats *out);
/* What the report records. */
size_t tp_shard_bytes(const Tp *tp, int socket);
void tp_free(Tp *tp);

#endif
