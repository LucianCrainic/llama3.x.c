#ifndef LLAMA_SHARDS_H
#define LLAMA_SHARDS_H

#include "runtime.h"

/* Tensor manifest: one descriptor per checkpoint tensor, built once after
 * checkpoint_map has validated the file. Every weight view (mapped, owned,
 * sharded or packed) is derived from these offsets, so all of them describe the
 * same tensors the same way. Offsets are bytes from the start of the file. */
typedef enum { DT_F32, DT_Q8 } TensorDtype;
typedef enum {
  T_EMBED,
  T_RMS_ATT,
  T_WQ,
  T_WK,
  T_WV,
  T_WO,
  T_RMS_FFN,
  T_W1,
  T_W2,
  T_W3,
  T_RMS_FINAL,
  T_ROPE_LEGACY,
  T_WCLS,
  T_ROLE_COUNT
} TensorRole;

typedef struct {
  TensorRole role;
  int layer;      /* -1 for global tensors */
  int rows, cols; /* W[out][in]: rows = out, cols = in; norms are 1 x dim */
  TensorDtype dtype;
  int group_size;      /* 0 for F32 */
  size_t offset;       /* values */
  size_t scale_offset; /* Q8 scales, else 0 */
  size_t bytes;        /* values + scales; an alias repeats its target's bytes without occupying new ones */
  int alias_of;        /* index of the shared tensor (tied wcls), else -1 */
} TensorDesc;

typedef struct TensorManifest {
  TensorDesc *t; /* in file order */
  int count;
  size_t file_bytes; /* end of the last non-alias tensor */
} TensorManifest;

/* Describes the layout of both executables: FP32 (`run`, formats 0/2/3) and Q8_0
 * (`runq`, formats 2/3). `header_bytes` is where the first tensor starts. */
void manifest_build(TensorManifest *m, const Config *p, int quantized, int gs, size_t header_bytes);
const TensorDesc *manifest_find(const TensorManifest *m, TensorRole role, int layer);
void manifest_free(TensorManifest *m);
const char *manifest_role_name(TensorRole role);

/* ---- Shards (B1) --------------------------------------------------------------------------------------------------
 * One socket's slice of each tensor, as a descriptor plus a place in that socket's weights region. S = 1 is the same
 * code: every weight is a ROWS shard with the full range, so tracks A, B and C share one path.
 *   ROWS    contiguous rows [row0, row0 + rows) of the source: one read of the values, one of the scales;
 *   COLS    columns [col0, col0 + cols) of every row (Wo, W2), packed to stride `cols` through bounded scratch; for Q8
 *           the matching group scales [col0 / gs, (col0 + cols) / gs) travel with them;
 *   COPY    the whole tensor on every socket (norm weights, always FP32);
 *   GLOBAL  one copy on the coordinator (an untied input embedding), kept outside the socket regions.
 * A tied embedding/classifier is the classifier's ROWS shards with `alias` set: socket s owns vocabulary rows
 * [row0, row0 + rows) for both the logits and the embedding lookup. The last socket takes a vocabulary remainder.
 * Quantized values and scales are never re-quantized: a group size that does not divide a socket's slice is rejected.
 */
enum { SHARD_PACKING_VERSION = 1 };
typedef enum { SHARD_ROWS, SHARD_COLS, SHARD_COPY, SHARD_GLOBAL } ShardKind;

typedef struct {
  const TensorDesc *src; /* valid as long as the manifest */
  ShardKind kind;
  int socket;
  int row0, rows, col0, cols; /* in the source tensor */
  int alias;                  /* a ROWS shard of the tied classifier, which is also the embedding table */
  size_t q_offset, s_offset;  /* bytes from the start of the shard's area (socket region, or the global area) */
  void *q;                    /* values: F32 or int8, rows x cols, stride ld */
  float *s;                   /* Q8 group scales, rows x (cols / gs); NULL for F32 */
  size_t ld;                  /* destination row stride in elements (= cols) */
} Shard;

typedef struct {
  int sockets, count, gs, quantized;
  Shard *t;
  size_t region_bytes[2]; /* what each socket's weights region must hold, in 64-byte aligned order */
  size_t global_bytes;    /* the GLOBAL area (coordinator DDR) */
  size_t scratch_limit;   /* COLS block scratch in bytes (default 64 MiB; at least one source row is always read) */
  size_t scratch_peak[2]; /* largest scratch each socket's load allocated */
  double rows_seconds[2], cols_seconds[2]; /* load time per socket, measured by shards_load */
} ShardSet;

/* Validates and plans. `_try` returns -1 with a message (incompatible heads, hidden size, or group size); the plain
 * form calls fail(). `gs` is the Q8 group size and is ignored for FP32 manifests. */
int shards_plan_try(ShardSet *set, const TensorManifest *m, const Config *p, int sockets, int gs, char *err,
                    size_t err_size);
void shards_plan(ShardSet *set, const TensorManifest *m, const Config *p, int sockets, int gs);
void shards_free(ShardSet *set);
/* The shard of (role, layer) on `socket`; NULL when the role is not sharded there. */
const Shard *shards_find(const ShardSet *set, TensorRole role, int layer, int socket);
/* Points every shard's q and s into the areas: socket_base[s] for socket s (set->region_bytes[s] bytes) and
 * global_base for GLOBAL shards (set->global_bytes). Bases must be 64-byte aligned. */
void shards_bind(ShardSet *set, void *const socket_base[2], void *global_base);
/* Reads the shards of `socket` from the checkpoint into their bound places. `team` (the socket's team) splits the
 * reads and the column packing among its workers; NULL runs on the calling thread. GLOBAL shards load with socket 0. */
struct Team;
void shards_load(ShardSet *set, int fd, int socket, struct Team *team);

#endif
