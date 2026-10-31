#include "runtime.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
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
void *memory_alloc(size_t count, size_t size, MemoryKind kind) {
  size_t bytes = checked_mul(count, size);
  if (bytes > PTRDIFF_MAX)
    fail("allocation exceeds addressable object size: %zu bytes", bytes);
  Allocation *block = calloc(1, checked_add(sizeof(*block), bytes));
  if (!block)
    fail("allocation failed for category %d: %zu bytes (%s)", kind, bytes, strerror(errno));
  block->allocation.bytes = bytes;
  block->allocation.kind = kind;
  memory_current[kind] = checked_add(memory_current[kind], bytes);
  if (memory_current[kind] > memory_peak[kind])
    memory_peak[kind] = memory_current[kind];
  total_current = checked_add(total_current, bytes);
  if (total_current > memory_total_peak)
    memory_total_peak = total_current;
  return block + 1;
}
void memory_free(void *pointer) {
  if (!pointer)
    return;
  Allocation *block = (Allocation *)pointer - 1;
  memory_current[block->allocation.kind] -= block->allocation.bytes;
  total_current -= block->allocation.bytes;
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
  if (quantized && (gs <= 0 || gs > 131071 || p->dim % gs || p->hidden_dim % gs))
    fail("unsupported INT8 group size %d: must divide dim and hidden_dim and fit int32 accumulation", gs);
}

static size_t tensor_bytes(size_t count, int quantized, int gs) {
  if (!quantized)
    return checked_mul(count, sizeof(float));
  /* Scales follow raw bytes with no padding in v2. Reject unaligned files. */
  if (count % (size_t)gs || count % _Alignof(float))
    fail("unsupported INT8 tensor alignment/group size: %zu elements", count);
  return checked_add(count, checked_mul(count / (size_t)gs, sizeof(float)));
}

void checkpoint_map(const char *path, int quantized, Config *p, int *shared, int *gs, int *fd, float **data,
                    size_t *bytes) {
  _Static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(Config) == 28,
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
  unsigned char header[256];
  size_t header_size = quantized ? 256 : 28;
  ssize_t got = pread(*fd, header, header_size, 0);
  if (got != (ssize_t)header_size)
    fail("cannot read checkpoint header '%s': %s", path, got < 0 ? strerror(errno) : "truncated read");
  uint32_t magic;
  memcpy(&magic, header, 4);
  if (quantized) {
    int version;
    memcpy(&version, header + 4, 4);
    if (magic != 0x616b3432 || version != 2)
      fail("unsupported INT8 checkpoint '%s': expected ak42 version 2", path);
    memcpy(p, header + 8, sizeof(*p));
    *shared = header[36];
    memcpy(gs, header + 37, 4);
    if (*shared > 1)
      fail("invalid classifier flag in '%s'", path);
  } else {
    if (magic == 0x616b3432)
      fail("unsupported FP32 checkpoint '%s': use legacy --version 0 export", path);
    memcpy(p, header, sizeof(*p));
    if (p->vocab_size == INT_MIN)
      fail("invalid vocabulary size in '%s'", path);
    *shared = p->vocab_size > 0;
    p->vocab_size = abs(p->vocab_size);
    *gs = 0;
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
  if (!quantized)
    needed = checked_add(needed, checked_mul(checked_mul((size_t)p->seq_len, d / p->n_heads), sizeof(float)));
  if (!*shared)
    needed = checked_add(needed, tensor_bytes(embedding, quantized, *gs));
  if (needed > *bytes)
    fail("truncated checkpoint '%s': need %zu bytes, found %zu", path, needed, *bytes);
  if (needed != *bytes)
    LOG(LOG_WARN, "checkpoint '%s' has %zu trailing bytes", path, *bytes - needed);
  /* Validate allocation extents too, before making the read-only mapping. */
  checked_mul(checked_mul(checked_mul(l, (size_t)p->seq_len), kv), sizeof(float));
  checked_mul(checked_mul((size_t)p->n_heads, (size_t)p->seq_len), sizeof(float));
  *data = mmap(NULL, *bytes, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED)
    fail("cannot map checkpoint '%s' (%zu bytes): %s", path, *bytes, strerror(errno));
  LOG(LOG_DEBUG, "mapped %zu checkpoint bytes; validated %zu", *bytes, needed);
}
