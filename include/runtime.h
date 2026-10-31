#ifndef LLAMA_RUNTIME_H
#define LLAMA_RUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* On-disk v0/v2 architecture. No implicit padding or extended RoPE metadata. */
typedef struct {
  int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
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
double monotonic_seconds(void);

/* Validation precedes pointer construction. The mapping owns no heap weights. */
void checkpoint_map(const char *path, int quantized, Config *config, int *shared, int *group_size, int *fd,
                    float **data, size_t *bytes);
void rmsnorm(float *out, const float *x, const float *weight, int size);
void softmax(float *x, int size);

typedef struct {
  Config config;
  void *implementation;
  size_t mapped_bytes;
} Model;
void engine_load(Model *model, const char *path);
float *engine_forward(Model *model, int token, int position);
void engine_free(Model *model);
const char *engine_precision(void);
int engine_group_size(void);

typedef struct {
  char *str;
  int id;
} TokenIndex;
typedef struct {
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512];
} Tokenizer;
void build_tokenizer(Tokenizer *t, const char *path, int vocab_size);
void free_tokenizer(Tokenizer *t);
const char *decode(Tokenizer *t, int token);
void safe_printf(const char *piece);
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
