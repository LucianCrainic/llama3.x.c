/* Profiled variant of native_c.c (docs/ttft-70b-diagnosis.md): the same id arrays through engine_forward_batch, one warm-up and three
 * measured requests per workload, with the engine's phase profile (LLAMA_PROFILE machinery, switched on through engine_set_profiling)
 * reset before and printed after each request's prefill. The per-phase seconds are therefore those of a warm process on exactly the
 * fixture ids. Usage: native_profile MODEL SHORT_IDS LONG_IDS CAPACITY BATCH WORKLOADS GENERATION
 *   WORKLOADS is short, long or both; GENERATION is the number of greedy tokens after the first (the decode is not profiled). */
#include "profile.h"
#include "runtime.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static void requests(Model *model, const char *name, const char *ids_file, int capacity, int batch, int generation) {
  FILE *input = fopen(ids_file, "r");
  int *ids = malloc((size_t)capacity * sizeof(int));
  if (!input || !ids)
    exit(2);
  int token, count = 0, status;
  while ((status = fscanf(input, "%d", &token)) == 1) {
    if (count + generation > capacity || token < 0 || token >= model->config.vocab_size)
      exit(2);
    ids[count++] = token;
  }
  fclose(input);
  if (status != EOF || count < 1)
    exit(2);
  int vocab = model->config.vocab_size;
  for (int repeat = 0; repeat < 4; ++repeat) {
    const float *logits = NULL;
    int microbatches = 0;
    memset(&profile, 0, sizeof(profile));
    profile_stage = STAGE_PREFILL;
    uint64_t start_wall = now_ns(CLOCK_REALTIME);
    uint64_t start = now_ns(CLOCK_MONOTONIC);
    for (int pos = 0; pos < count;) {
      int rows = count - pos < batch ? count - pos : batch;
      logits = engine_forward_batch(model, ids + pos, rows, pos, pos + rows == count ? LOGITS_LAST : LOGITS_NONE);
      pos += rows;
      microbatches++;
    }
    int first_token = greedy(logits, vocab);
    uint64_t first = now_ns(CLOCK_MONOTONIC);
    printf("{\"workload\":\"%s\",\"repeat\":%d,\"warmup\":%s,\"prompt_tokens\":%d,\"microbatches\":%d,\"ttft_s\":%.9f,"
           "\"start_epoch_ns\":%" PRIu64 ",\"first_token\":%d,\"profile\":{",
           name, repeat, repeat ? "false" : "true", count, microbatches, (double)(first - start) / 1e9, start_wall, first_token);
    for (int phase = 0; phase < PHASE_COUNT; ++phase)
      printf("%s\"%s\":%.9g", phase ? "," : "", profile_phase_name((Phase)phase), profile.seconds[STAGE_PREFILL][phase]);
    puts("}}");
    fflush(stdout);
    /* generated tokens keep the cache state of the plain driver; their time is not reported */
    profile_stage = STAGE_DECODE;
    int last = first_token;
    for (int i = 1; i < generation; ++i)
      last = greedy(engine_forward(model, last, count + i - 1), vocab);
  }
  free(ids);
}

int main(int argc, char **argv) {
  if (argc != 8)
    return 2;
  Model model = {0};
  int capacity = atoi(argv[4]), batch = atoi(argv[5]), generation = atoi(argv[7]);
  if (capacity <= 0 || batch <= 0 || generation <= 0)
    return 2;
  engine_load_context(&model, argv[1], capacity);
  if (model.config.seq_len != capacity || engine_max_batch(&model) != batch)
    fail("Native benchmark context or batch differs from the requested setup");
  engine_set_profiling(&model, 1);
  if (strcmp(argv[6], "long"))
    requests(&model, "short", argv[2], capacity, batch, generation);
  if (strcmp(argv[6], "short"))
    requests(&model, "long", argv[3], capacity, batch, generation);
  engine_free(&model);
  return 0;
}
