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

#endif
