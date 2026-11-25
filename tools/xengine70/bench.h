/* Shared in-process timing for the two native 70B engines. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
  void *ctx;
  const float *(*forward)(void *ctx, int *ids, int count, int pos, int logits);
  void (*reset)(void *ctx);
  int vocab, batch, capacity;
} Bench;

static uint64_t now_ns(clockid_t clock) {
  struct timespec ts;
  if (clock_gettime(clock, &ts))
    exit(2);
  return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static int greedy(const float *logits, int vocab) {
  if (!logits)
    exit(2);
  int best = 0;
  for (int i = 1; i < vocab; ++i)
    if (logits[i] > logits[best])
      best = i;
  return best;
}

static int requests(Bench *bench, const char *name, const char *ids_file) {
  enum { generation = 128 };
  FILE *input = fopen(ids_file, "r");
  int *ids = (int *)malloc((size_t)bench->capacity * sizeof(int));
  if (!input || !ids)
    exit(2);
  int token, count = 0, status;
  while ((status = fscanf(input, "%d", &token)) == 1) {
    if (count + generation > bench->capacity || token < 0 || token >= bench->vocab)
      exit(2);
    ids[count++] = token;
  }
  fclose(input);
  if (status != EOF || count < 1)
    exit(2);

  for (int repeat = 0; repeat < 4; ++repeat) {
    if (bench->reset)
      bench->reset(bench->ctx);
    int output[generation];
    const float *logits = NULL;
    uint64_t start_wall = now_ns(CLOCK_REALTIME);
    uint64_t start = now_ns(CLOCK_MONOTONIC);
    for (int pos = 0; pos < count;) {
      int rows = count - pos < bench->batch ? count - pos : bench->batch;
      logits = bench->forward(bench->ctx, ids + pos, rows, pos, pos + rows == count);
      pos += rows;
    }
    output[0] = greedy(logits, bench->vocab);
    uint64_t first = now_ns(CLOCK_MONOTONIC);
    for (int i = 1; i < generation; ++i) {
      logits = bench->forward(bench->ctx, &output[i - 1], 1, count + i - 1, 1);
      output[i] = greedy(logits, bench->vocab);
    }
    uint64_t end = now_ns(CLOCK_MONOTONIC);
    printf("{\"workload\":\"%s\",\"repeat\":%d,\"warmup\":%s,\"prompt_tokens\":%d,\"generated_tokens\":%d,"
           "\"ttft_s\":%.9f,\"decode_s\":%.9f,\"start_epoch_ns\":%" PRIu64 ",\"token_ids\":[",
           name, repeat, repeat ? "false" : "true", count, generation,
           (double)(first - start) / 1e9, (double)(end - first) / 1e9, start_wall);
    for (int i = 0; i < generation; ++i)
      printf("%s%d", i ? "," : "", output[i]);
    puts("]}");
    fflush(stdout);
  }
  free(ids);
  return count + generation - 1;
}
