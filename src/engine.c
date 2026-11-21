/* LLaMA 3 CPU inference, derived from Andrej Karpathy and James W. DeLancey. See LICENSE.
 *
 * The engine of both executables. run links it with engine_q8 = 0 (FP32 checkpoints, optional BF16 prefill on packed
 * weights); runq with engine_q8 = 1 (Q8_0 checkpoints, W8A8 or W8A32 activations). This file maps the checkpoint, picks
 * the prefill policy of each format and drives the shard engine (tp.c), which holds the one forward pass of both: on one
 * socket over whole-tensor shards viewed in the weight store (tp_create_local), on two tensor-parallel (tp_create). */
#include "loader.h"
#include "prefill.h"
#include "runtime.h"
#include "shards.h"
#include "tp.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* One model per CLI process. */
static int GS;               /* Q8: the checkpoint's group size; 0 for FP32 and for per-channel INT8 (format 4) */
static int fp32_activations; /* Q8: LLAMA_Q8_ACTIVATIONS=fp32 multiplies dequantized weights by FP32 activations (W8A32)
                                instead of quantizing activations (the default W8A8 Q8_0 rule) */
static int bf16_mode;        /* FP32: LLAMA_PRECISION=bf16 (C5) */

typedef struct {
  Config config;           // the hyperparameters of the architecture (the blueprint)
  TensorManifest manifest; // where each tensor lives in the file
  WeightStore store;       // one socket: where those bytes live now, the mapping or owned regions (A4)
  Tp *tp;                  // the shard engine: weights views, KV caches, activations and the forward pass
  float *batch_logits;     // per-token prefill with all-position logits: M x vocab, allocated on first use
  int bf16;                // BF16 mode
  // Mapping ownership: one socket's weights borrow this region in file mode until engine_free().
  int fd;           // file descriptor for memory mapping
  float *data;      // memory mapped data pointer
  size_t file_size; // size of the checkpoint file in bytes
} Transformer;

static void read_checkpoint(const char *path, Transformer *t) {
  int shared, gs;
  double start = monotonic_seconds();
  checkpoint_map(path, engine_q8, &t->config, &shared, &gs, &t->fd, &t->data, &t->file_size, &t->manifest);
  GS = engine_q8 ? gs : 0;
  metrics.checkpoint_s = monotonic_seconds() - start;
  /* Placement time is reported on its own, not as checkpoint_load. */
  if (tp_sockets_requested() == 1) /* two sockets load their shards in tp_create, straight from the file */
    weights_open(&t->store, t->fd, &t->data, t->file_size, &t->manifest);
  metrics.checkpoint_ready = 1;
}

static void free_transformer(Transformer *t) {
  tp_free(t->tp);
  memory_free(t->batch_logits);
  weights_close(&t->store);
  manifest_free(&t->manifest);
  // close the memory mapping
  if (t->data != MAP_FAILED)
    munmap(t->data, t->file_size);
  if (t->fd != -1)
    close(t->fd);
}

/* A row reads only its own activations and the KV rows of earlier positions, so one pass per microbatch (layer-major)
 * equals per-token execution: bitwise on the per-token kernels; with the batched GEMM, FP32 within the GEMM's rounding
 * and Q8 (W8A8 only) within the Q8 budget, bitwise on the scalar instruction set. With the GEMM on, microbatches of more
 * than one row run batched; in BF16 mode every microbatch does, even a single row, so a prompt's precision does not
 * depend on how it was split. Otherwise (one row, W8A32, or the GEMM switched off) the rows run per token. */
static float *forward_batch(Model *model, const int *tokens, int count, int pos0, LogitsWanted want) {
  prefill_validate(model, tokens, count, pos0, want);
  Transformer *t = model->implementation;
  if (model->prefill_gemm && (count > 1 || t->bf16))
    return tp_forward_batch(t->tp, tokens, count, pos0, want);
  size_t vocab = (size_t)t->config.vocab_size;
  if (want == LOGITS_ALL && !t->batch_logits)
    t->batch_logits = memory_alloc(checked_mul((size_t)model->max_batch, vocab), sizeof(float), MEM_ACTIVATION);
  float *logits = NULL;
  for (int i = 0; i < count; i++) {
    /* The classifier runs only for the rows whose logits were asked for. */
    int wanted = want == LOGITS_ALL || (want == LOGITS_LAST && i == count - 1);
    logits = tp_forward(t->tp, tokens[i], pos0 + i, wanted);
    if (want == LOGITS_ALL)
      memcpy(t->batch_logits + (size_t)i * vocab, logits, vocab * sizeof(float));
  }
  return want == LOGITS_NONE ? NULL : want == LOGITS_ALL ? t->batch_logits : logits;
}

/* ---- load: the prefill policy of each format (D2 §7) ------------------------------------------------------------- */

/* FP32: 128 rows per pass on one socket, 64 on two, 256 for the AMX-BF16 rung (LLAMA_PRECISION=bf16). */
static void configure_fp32(Model *model, Transformer *t) {
  model->prefill_bf16 = prefill_precision(0);
  prefill_configure(model, model->prefill_bf16 ? 256 : tp_sockets_requested() == 2 ? 64 : 128);
  bf16_mode = t->bf16 = model->prefill_bf16;
  model->prefill_kernel = t->bf16 ? (kernels_amx_active() ? "amx-bf16" : "bf16-reference") : kernels_name();
}

/* Q8: W8A8 prefill takes 256 rows per pass, on AMX-INT8 where it can run (C6); W8A32 never batches, it stays on the
 * per-row kernel. The rungs below AMX: the VNNI Q8 GEMM, and the portable one when the group size is not a multiple of
 * 64. */
static void configure_q8(Model *model) {
  prefill_configure(model, fp32_activations ? 1 : 256);
  /* Per-channel INT8 (GS 0): each product's group is its per-socket input width, whole 64-byte k-steps when dim and
   * hidden_dim are multiples of 128 (one or two sockets). */
  const Config *c = &model->config;
  int gs64 = GS ? GS % 64 == 0 : c->dim % 128 == 0 && c->hidden_dim % 128 == 0;
  if (!fp32_activations && model->prefill_gemm && gs64)
    kernels_select_amx_auto();
  if (fp32_activations) {
    model->prefill_gemm = 0;
    if (kernels_amx_active())
      LOG(LOG_INFO, "W8A32 prefill uses avx512: the per-row kernel, never AMX-INT8");
  } else if (model->prefill_gemm) {
    if (kernels_amx_active() && gs64) {
      model->prefill_kernel = "amx-int8";
    } else if (kernels_avx512_active() && gs64) {
      model->prefill_kernel = "avx512-q8";
    } else {
      model->prefill_kernel = "portable-q8";
      if (kernels_avx512_active())
        LOG(LOG_INFO, "W8A8 batched prefill: group size %d is not a multiple of 64, using the portable Q8 GEMM", GS);
    }
    if (kernels_amx_active() && !gs64)
      LOG(LOG_INFO, "AMX-INT8 needs a group size that is a multiple of 64 (this checkpoint has %d): falling back", GS);
  }
}

/* CLI adapter: the executable picks the checkpoint format (engine_q8). */
void engine_load(Model *model, const char *path) {
  engine_load_context(model, path, 0);
}
void engine_load_context(Model *model, const char *path, int context) {
  Transformer *t = memory_alloc(1, sizeof(*t), MEM_METADATA);
  t->fd = -1;
  model->implementation = t;
  kernels_init();
  kernels_backend_init();
  if (engine_q8) {
    prefill_precision(1); /* LLAMA_PRECISION=bf16 is an error here */
    const char *activations = getenv("LLAMA_Q8_ACTIVATIONS");
    if (activations && *activations && strcmp(activations, "int8") && strcmp(activations, "fp32"))
      fail("LLAMA_Q8_ACTIVATIONS must be int8 or fp32");
    fp32_activations = activations && !strcmp(activations, "fp32");
  }
  read_checkpoint(path, t);
  model->config = t->config;
  model->file_bytes = t->file_size;
  model->mapped_bytes = t->data == MAP_FAILED ? 0 : t->file_size;
  if (context < 0 || context > t->config.seq_len)
    fail("requested context exceeds checkpoint context");
  t->config.seq_len = context ? context : (t->config.seq_len < 4096 ? t->config.seq_len : 4096);
  model->config = t->config;
  double start = monotonic_seconds();
  if (engine_q8)
    configure_q8(model);
  else
    configure_fp32(model, t);
  /* The packed copies the batched GEMM reads: BF16 (FP32 checkpoint, BF16 mode) or int8 for AMX-INT8. */
  int amx_int8 = engine_q8 && model->prefill_gemm && model->prefill_kernel && !strcmp(model->prefill_kernel, "amx-int8");
  TpPack pack = t->bf16 ? TP_PACK_BF16 : amx_int8 ? TP_PACK_INT8 : TP_PACK_NONE;
  int rows = model->prefill_gemm ? model->max_batch : 1;
  if (tp_sockets_requested() == 2) {
    /* D1: with the GEMM on, microbatches run through the batched two-socket forward, else per token. */
    t->tp = tp_create(&t->config, &t->manifest, t->fd, engine_q8, GS, fp32_activations, rows, pack);
    munmap(t->data, t->file_size); /* the shards were read with pread */
    t->data = MAP_FAILED;
    model->mapped_bytes = 0;
  } else {
    t->tp = tp_create_local(&t->config, &t->manifest, &t->store, engine_q8, GS, fp32_activations, rows, pack);
  }
  metrics.runtime_s = monotonic_seconds() - start;
  metrics.runtime_ready = 1;
}
void engine_kv_stats(Model *model, int position, int verify, KvStats *out) {
  Transformer *t = model->implementation;
  tp_kv_stats(t->tp, position, verify, out);
}
float *engine_forward(Model *model, int token, int position) {
  if (token < 0 || token >= model->config.vocab_size || position < 0 || position >= model->config.seq_len)
    fail("forward input out of range: token=%d position=%d", token, position);
  Transformer *t = model->implementation;
  float *logits = tp_forward(t->tp, token, position, 1);
  park_if_failing();
  return logits;
}
float *engine_forward_batch(Model *model, const int *tokens, int count, int pos0, LogitsWanted want) {
  float *logits = forward_batch(model, tokens, count, pos0, want);
  park_if_failing();
  return logits;
}
void engine_free(Model *model) {
  if (model->implementation) {
    free_transformer(model->implementation);
    memory_free(model->implementation);
    model->implementation = NULL;
    kernels_backend_free();
  }
}
const char *engine_precision(void) {
  if (engine_q8)
    return fp32_activations ? "int8_q8_0_w8a32" : "int8_q8_0";
  return bf16_mode ? "bf16_from_fp32" : "fp32";
}
int engine_group_size(void) {
  return GS;
}
