#include "runtime.h"

#include "shards.h"
#include "workers.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

LogLevel log_level = LOG_INFO;
char failure_message[512];
Metrics metrics;
size_t memory_current[MEM_COUNT], memory_peak[MEM_COUNT], memory_total_peak;
static size_t total_current;

typedef union {
  max_align_t alignment;
  struct {
    size_t bytes;
    MemoryKind kind;
  } allocation;
} Allocation;

void log_message(LogLevel level, const char *format, ...) {
  static const char *names[] = {"ERROR", "WARN", "INFO", "DEBUG", "TRACE"};
  fprintf(stderr, "[%s] ", names[level]);
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fputc('\n', stderr);
}

_Noreturn void fail(const char *format, ...) {
  /* Attention heads can discover the same corrupt input concurrently. Exactly
   * one thread owns fatal diagnostics and exit handlers; other failing workers
   * wait for process termination instead of truncating the report again. */
  static atomic_flag fatal_started = ATOMIC_FLAG_INIT;
  if (atomic_flag_test_and_set_explicit(&fatal_started, memory_order_relaxed)) {
    for (;;)
      pause();
  }
  workers_abort(); /* wait loops must not spin on memory the exit handlers release */
  va_list args;
  va_start(args, format);
  vsnprintf(failure_message, sizeof(failure_message), format, args);
  va_end(args);
  log_message(LOG_ERROR, "%s", failure_message);
  exit(EXIT_FAILURE);
}

size_t checked_mul(size_t a, size_t b) {
  if (a && b > SIZE_MAX / a)
    fail("size multiplication overflow: %zu * %zu", a, b);
  return a * b;
}
size_t checked_add(size_t a, size_t b) {
  if (b > SIZE_MAX - a)
    fail("size addition overflow: %zu + %zu", a, b);
  return a + b;
}
void memory_track(MemoryKind kind, size_t bytes) {
  memory_current[kind] = checked_add(memory_current[kind], bytes);
  if (memory_current[kind] > memory_peak[kind])
    memory_peak[kind] = memory_current[kind];
  total_current = checked_add(total_current, bytes);
  if (total_current > memory_total_peak)
    memory_total_peak = total_current;
}
void memory_untrack(MemoryKind kind, size_t bytes) {
  memory_current[kind] -= bytes;
  total_current -= bytes;
}
void *memory_alloc(size_t count, size_t size, MemoryKind kind) {
  size_t bytes = checked_mul(count, size);
  if (bytes > PTRDIFF_MAX)
    fail("allocation exceeds addressable object size: %zu bytes", bytes);
  Allocation *block = calloc(1, checked_add(sizeof(*block), bytes));
  if (!block)
    fail("allocation failed for category %d: %zu bytes (%s)", kind, bytes, strerror(errno));
  block->allocation.bytes = bytes;
  block->allocation.kind = kind;
  memory_track(kind, bytes);
  return block + 1;
}
void memory_free(void *pointer) {
  if (!pointer)
    return;
  Allocation *block = (Allocation *)pointer - 1;
  memory_untrack(block->allocation.kind, block->allocation.bytes);
  free(block);
}

double monotonic_seconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts))
    fail("monotonic clock failed: %s", strerror(errno));
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void validate_config(Config *p, int quantized, int gs) {
  if (p->dim <= 0 || p->hidden_dim <= 0 || p->n_layers <= 0 || p->n_heads <= 0 || p->n_kv_heads <= 0 ||
      p->vocab_size <= 0 || p->seq_len <= 0)
    fail("invalid checkpoint dimensions: all seven dimensions must be positive");
  if (p->dim % p->n_heads || p->n_heads % p->n_kv_heads || (p->dim / p->n_heads) % 2)
    fail("unsupported attention dimensions: dim must be divisible by heads, heads must be divisible by KV heads, head "
         "size must be even");
  if (!isfinite(p->norm_eps) || p->norm_eps <= 0 || !isfinite(p->rope_theta) || p->rope_theta <= 1 ||
      p->head_dim != p->dim / p->n_heads || p->max_seq_len != p->seq_len)
    fail("invalid RMSNorm, RoPE theta, head dimension or context metadata");
  if (p->rope_type != 0 && p->rope_type != 1)
    fail("unsupported RoPE scaling type %d", p->rope_type);
  if (!isfinite(p->rope_factor) || !isfinite(p->rope_low) || !isfinite(p->rope_high) || p->rope_factor < 1 ||
      p->rope_low <= 0 || p->rope_high <= p->rope_low || p->rope_original <= 0)
    fail("invalid RoPE scaling metadata");
  if (p->format == 3 && (p->bos_id < 0 || p->bos_id >= p->vocab_size || p->eos_id < 0 || p->eos_id >= p->vocab_size ||
                         p->eot_id < -1 || p->eot_id >= p->vocab_size || p->eom_id < -1 || p->eom_id >= p->vocab_size))
    fail("invalid special token metadata");
  if (quantized && (gs <= 0 || gs > 131071 || p->dim % gs || p->hidden_dim % gs))
    fail("unsupported INT8 group size %d: must divide dim and hidden_dim and fit int32 accumulation", gs);
}

size_t tensor_bytes(size_t count, int quantized, int gs) {
  if (!quantized)
    return checked_mul(count, sizeof(float));
  /* Scales follow raw bytes with no padding in v2. Reject unaligned files. */
  if (count % (size_t)gs || count % _Alignof(float))
    fail("unsupported INT8 tensor alignment/group size: %zu elements", count);
  return checked_add(count, checked_mul(count / (size_t)gs, sizeof(float)));
}

void checkpoint_map(const char *path, int quantized, Config *p, int *shared, int *gs, int *fd, float **data,
                    size_t *bytes, TensorManifest *manifest) {
  _Static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(int32_t) == 4 && offsetof(Config, format) == 28,
                 "checkpoint ABI requires 32-bit int/float and seven packed ints");
  const uint32_t endian = 1;
  if (*(const unsigned char *)&endian != 1)
    fail("only little-endian checkpoints are supported");
  *fd = open(path, O_RDONLY);
  if (*fd < 0)
    fail("cannot open checkpoint '%s': %s", path, strerror(errno));
  struct stat st;
  if (fstat(*fd, &st) || !S_ISREG(st.st_mode) || st.st_size < (quantized ? 256 : 28)) {
    close(*fd);
    *fd = -1;
    fail("invalid/truncated checkpoint '%s': expected a regular file with complete header", path);
  }
  if ((uintmax_t)st.st_size > SIZE_MAX || (uintmax_t)st.st_size > PTRDIFF_MAX)
    fail("checkpoint '%s' is too large for this address space", path);
  *bytes = (size_t)st.st_size;
  memset(p, 0, sizeof(*p));
  p->norm_eps = 1e-5f;
  p->rope_theta = ROPE_THETA;
  p->rope_factor = 1;
  p->rope_low = 1;
  p->rope_high = 4;
  p->rope_original = 8192;
  p->bos_id = BOS_ID;
  p->eos_id = EOS_ID;
  p->eot_id = EOT_ID;
  p->eom_id = -1;
  unsigned char header[256];
  size_t header_size = quantized ? 256 : 28;
  ssize_t got = pread(*fd, header, header_size, 0);
  if (got != (ssize_t)header_size)
    fail("cannot read checkpoint header '%s': %s", path, got < 0 ? strerror(errno) : "truncated read");
  uint32_t magic;
  memcpy(&magic, header, 4);
  if (magic == 0x616b3432) {
    if (*bytes < 256 || pread(*fd, header, 256, 0) != 256)
      fail("truncated versioned checkpoint header");
    header_size = 256;
    int version;
    memcpy(&version, header + 4, 4);
    if (version == 3) {
      int precision;
      memcpy(&precision, header + 36, 4);
      if (precision < 0 || precision > 1)
        fail("unsupported checkpoint precision %d", precision);
      if (precision != quantized)
        fail("checkpoint precision requires %s executable", precision == 0 ? "run" : "runq");
      memcpy(p, header + 8, 28);
      memcpy(shared, header + 40, 4);
      memcpy(gs, header + 44, 4);
      memcpy(&p->norm_eps, header + 48, 4);
      memcpy(&p->rope_theta, header + 52, 4);
      memcpy(&p->rope_type, header + 56, 4);
      memcpy(&p->rope_factor, header + 60, 4);
      memcpy(&p->rope_low, header + 64, 4);
      memcpy(&p->rope_high, header + 68, 4);
      memcpy(&p->rope_original, header + 72, 4);
      memcpy(&p->max_seq_len, header + 76, 4);
      memcpy(&p->head_dim, header + 80, 4);
      memcpy(&p->bos_id, header + 84, 4);
      memcpy(&p->eos_id, header + 88, 4);
      memcpy(&p->eot_id, header + 92, 4);
      memcpy(&p->eom_id, header + 96, 4);
      memcpy(p->name, header + 100, 128);
      if (!memchr(p->name, 0, 128))
        fail("unterminated model name");
      for (int i = 228; i < 256; i++)
        if (header[i])
          fail("unsupported nonzero reserved checkpoint metadata");
      if (!quantized && *gs != 0)
        fail("FP32 checkpoint must have group size zero");
    } else if (version == 2 && quantized) {
      memcpy(p, header + 8, 28);
      *shared = header[36];
      memcpy(gs, header + 37, 4);
    } else
      fail("unsupported checkpoint version %d for this executable", version);
    p->format = version;
    if (*shared < 0 || *shared > 1)
      fail("invalid classifier flag in '%s'", path);
  } else {
    if (quantized)
      fail("unsupported INT8 checkpoint: expected ak42 version 2 or 3");
    memcpy(p, header, 28);
    if (p->vocab_size == INT_MIN)
      fail("invalid vocabulary size in '%s'", path);
    *shared = p->vocab_size > 0;
    p->vocab_size = abs(p->vocab_size);
    *gs = 0;
  }
  p->shared = *shared;
  if (p->format != 3) {
    p->max_seq_len = p->seq_len;
    p->head_dim = p->n_heads > 0 ? p->dim / p->n_heads : 0;
  }
  validate_config(p, quantized, *gs);
  size_t d = p->dim, h = p->hidden_dim, l = p->n_layers;
  size_t kv = d / (size_t)p->n_heads * (size_t)p->n_kv_heads;
  size_t embedding = checked_mul((size_t)p->vocab_size, d);
  size_t norm = checked_mul(checked_add(checked_mul(2, l), 1), d);
  size_t needed = checked_add(header_size, checked_mul(norm, sizeof(float)));
  needed = checked_add(needed, tensor_bytes(embedding, quantized, *gs));
  size_t shapes[] = {checked_mul(d, d), checked_mul(d, kv), checked_mul(d, kv), checked_mul(d, d),
                     checked_mul(d, h), checked_mul(h, d),  checked_mul(d, h)};
  for (size_t i = 0; i < 7; i++)
    needed = checked_add(needed, checked_mul(l, tensor_bytes(shapes[i], quantized, *gs)));
  if (!quantized && p->format == 0)
    needed = checked_add(needed, checked_mul(checked_mul((size_t)p->seq_len, d / p->n_heads), sizeof(float)));
  if (!*shared)
    needed = checked_add(needed, tensor_bytes(embedding, quantized, *gs));
  if (needed > *bytes)
    fail("truncated checkpoint '%s': need %zu bytes, found %zu", path, needed, *bytes);
  if (p->format == 3 && needed != *bytes)
    fail("version 3 checkpoint has unexpected trailing bytes");
  if (needed != *bytes)
    LOG(LOG_WARN, "checkpoint '%s' has %zu trailing bytes", path, *bytes - needed);
  /* The manifest is an independent description of the same layout; any
   * disagreement with the arithmetic above is a bug, never a warning. */
  manifest_build(manifest, p, quantized, *gs, header_size);
  if (manifest->file_bytes != needed)
    fail("tensor manifest covers %zu bytes but the checkpoint layout needs %zu", manifest->file_bytes, needed);
  /* Validate allocation extents too, before making the read-only mapping. */
  checked_mul(checked_mul(checked_mul(l, (size_t)p->seq_len), kv), sizeof(float));
  checked_mul(checked_mul((size_t)p->n_heads, (size_t)p->seq_len), sizeof(float));
  *data = mmap(NULL, *bytes, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED)
    fail("cannot map checkpoint '%s' (%zu bytes): %s", path, *bytes, strerror(errno));
  LOG(LOG_DEBUG, "mapped %zu checkpoint bytes; validated %zu", *bytes, needed);
}
