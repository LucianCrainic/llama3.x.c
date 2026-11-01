#ifndef LLAMA_RUNTIME_H
#define LLAMA_RUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* First seven fields are the legacy disk dimensions; never serialize sizeof(Config). */
typedef struct {
  int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
  int format, max_seq_len, shared, head_dim;
  float norm_eps, rope_theta, rope_factor, rope_low, rope_high;
  int rope_type, rope_original;
  int bos_id, eos_id, eot_id, eom_id;
  char name[128];
} Config;

enum { BOS_ID = 128000, EOS_ID = 128001, EOT_ID = 128009 };
#define ROPE_THETA 500000.0f

typedef enum { LOG_ERROR, LOG_WARN, LOG_INFO, LOG_DEBUG, LOG_TRACE } LogLevel;
extern LogLevel log_level;
void log_message(LogLevel level, const char *format, ...);
#define LOG(level, ...)                                                                                                \
  do {                                                                                                                 \
    if ((level) <= log_level)                                                                                          \
      log_message(level, __VA_ARGS__);                                                                                 \
  } while (0)
_Noreturn void fail(const char *format, ...);
extern char failure_message[512];

/* Allocator categories are the future memory-placement boundary. Mapped weights
 * are tracked separately. Allocations are zeroed and owned by their caller. */
typedef enum {
  MEM_KV,
  MEM_ACTIVATION,
  MEM_EMBEDDING,
  MEM_METADATA,
  MEM_TOKENIZER,
  MEM_SAMPLER,
  MEM_PROMPT,
  MEM_COUNT
} MemoryKind;
extern size_t memory_current[MEM_COUNT], memory_peak[MEM_COUNT], memory_total_peak;
void *memory_alloc(size_t count, size_t size, MemoryKind kind);
void memory_free(void *pointer);
size_t checked_mul(size_t a, size_t b);
size_t checked_add(size_t a, size_t b);
/* File bytes of a tensor: FP32 values, or Q8 values plus FP32 group scales (no padding). */
size_t tensor_bytes(size_t count, int quantized, int gs);
double monotonic_seconds(void);

/* Validation precedes pointer construction. The mapping owns no heap weights;
 * the caller owns `manifest` (manifest_free) once the call succeeds. */
struct TensorManifest;
void checkpoint_map(const char *path, int quantized, Config *config, int *shared, int *group_size, int *fd,
                    float **data, size_t *bytes, struct TensorManifest *manifest);
void rmsnorm(float *out, const float *x, const float *weight, int size);
void rmsnorm_eps(float *out, const float *x, const float *weight, int size, float epsilon);
float rope_frequency(const Config *config, int pair_dimension);
void apply_rope(float *query, float *key, const Config *config, int position);
void softmax(float *x, int size);

typedef struct {
  Config config;
  void *implementation;
  size_t mapped_bytes;
} Model;
void engine_load(Model *model, const char *path);
void engine_load_context(Model *model, const char *path, int context);
float *engine_forward(Model *model, int token, int position);
void engine_free(Model *model);
const char *engine_precision(void);
int engine_group_size(void);

typedef struct {
  char *str;
  int id;
  size_t length;
} TokenIndex;
typedef struct {
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  int version, base_tokens, bos_id, eos_id, eot_id, eom_id, start_header_id, end_header_id;
  int *lengths;
  void *regex;
  char *chat_prefix;
  unsigned int max_token_length;
  unsigned char byte_pieces[512];
} Tokenizer;
void build_tokenizer(Tokenizer *t, const char *path, int vocab_size);
void free_tokenizer(Tokenizer *t);
void tokenizer_check_model(const Tokenizer *t, const Config *p);
int tokenizer_stop(const Tokenizer *t, int token);
int *chat_prompt(Tokenizer *t, const char *system, const char *user, int first, int *count);
const char *decode(Tokenizer *t, int token);
void safe_printf(const char *piece);
void print_token(Tokenizer *t, int token);
/* tokens must hold strlen(text)+2 elements (at most one token per byte + BOS/EOS). */
void encode(Tokenizer *t, const char *text, int8_t bos, int8_t eos, int *tokens, int *count);

typedef struct {
  float prob;
  int index;
} ProbIndex;
typedef struct {
  int vocab_size;
  ProbIndex *probindex;
  float temperature, topp;
  unsigned long long rng_state;
} Sampler;
void build_sampler(Sampler *s, int vocab_size, float temperature, float topp, unsigned long long seed);
void free_sampler(Sampler *s);
int sample(Sampler *s, float *logits);
int sample_argmax(float *values, int n);
float random_f32(unsigned long long *state);

typedef struct {
  double checkpoint_s, tokenizer_s, runtime_s, tokenization_s, prefill_s, decode_s;
  double inference_s, end_to_end_s, ttft_s;
  size_t input_tokens, processed_tokens, output_tokens, forward_passes, decode_steps;
  size_t sampled_tokens, stop_tokens, turns;
  int checkpoint_ready, tokenizer_ready, runtime_ready, inference_started;
  int has_first_token, completed;
  const char *stop_reason;
} Metrics;
extern Metrics metrics;

#endif
