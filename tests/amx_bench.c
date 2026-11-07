/* AMX tile microbenchmark (C4): the BF16 and int8 2 x 2 tile microkernels on operands that stay in L2, one private
 * working set per worker, so the number is the tile engine's, not the memory system's. One JSON object per line.
 *   amx-bench [--kind bf16|int8|both] [--k K] [--reps R]
 * Threads, pinning and the instruction set come from the engine's environment, as for gemm-bench:
 * LLAMA_KERNELS=amx LLAMA_BACKEND=teams LLAMA_THREADS=1|56 under numactl --cpunodebind=0. Compare the result with
 * the measured peaks of docs/hbm-roofline.md (112 TFLOP/s BF16 and 225 TOPS int8 for 56 cores). */
#include "gemm.h"
#include "kernels.h"
#include "workers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int bf16, k, reps, workers;
  void **base; /* per worker: A panel, packed B, C */
} Job;

enum { M = 32, N = 32 };

static size_t round_up(size_t v, size_t to) {
  return (v + to - 1) / to * to;
}

static void prepare(const void *p, int worker, int workers) {
  const Job *job = p;
  (void)workers;
  size_t elem = job->bf16 ? 2 : 1, kp = round_up((size_t)job->k, job->bf16 ? 32 : 64);
  size_t bytes = (size_t)M * kp * elem + (size_t)N * kp * elem + (size_t)M * N * 4 + 192;
  char *buffer = aligned_alloc(64, round_up(bytes, 64));
  if (!buffer)
    fail("cannot allocate the benchmark operands");
  memset(buffer, 0, round_up(bytes, 64));
  /* Small deterministic values: finite, and the int8 sums cannot overflow int32. */
  for (size_t i = 0; i < (size_t)(M + N) * kp * elem; i++)
    buffer[i] = job->bf16 ? (char)((i % 2) ? 0x3f : (i * 7) & 0x7f) : (char)((int)(i * 13 % 7) - 3);
  job->base[worker] = buffer;
}

static void run(const void *p, int worker, int workers) {
  const Job *job = p;
  (void)workers;
  size_t elem = job->bf16 ? 2 : 1, kp = round_up((size_t)job->k, job->bf16 ? 32 : 64);
  char *a = job->base[worker], *b = a + (size_t)M * kp * elem, *c = b + (size_t)N * kp * elem;
  /* B is already in the packed layout: any bytes are valid operands for a throughput test. */
  for (int r = 0; r < job->reps; r++) {
    if (job->bf16)
      amx_bf16_tile_gemm((float *)c, N, (const uint16_t *)a, kp, (const uint16_t *)b, M, N, job->k);
    else
      amx_s8_tile_gemm((int32_t *)c, N, (const int8_t *)a, kp, (const int8_t *)b, M, N, job->k);
  }
}

static void bench(int bf16, int k, int reps) {
  Team *team = pool_team(0);
  int workers = team ? team_size(team) : 1;
  Job job = {.bf16 = bf16, .k = k, .reps = reps, .workers = workers, .base = calloc((size_t)workers, sizeof(void *))};
  if (team) {
    team_run(team, prepare, &job);
    team_run(team, run, &job); /* warm-up */
  } else {
    prepare(&job, 0, 1);
    run(&job, 0, 1);
  }
  double best = 1e30;
  for (int trial = 0; trial < 5; trial++) {
    double start = monotonic_seconds();
    if (team)
      team_run(team, run, &job);
    else
      run(&job, 0, 1);
    double elapsed = monotonic_seconds() - start;
    best = elapsed < best ? elapsed : best;
  }
  double ops = 2.0 * M * N * k * reps * workers;
  printf("{\"kind\": \"%s\", \"isa\": \"%s\", \"backend\": \"%s\", \"workers\": %d, \"k\": %d, \"reps\": %d, "
         "\"best_s\": %.9f, \"tera_ops\": %.3f, \"tera_ops_per_worker\": %.4f}\n",
         bf16 ? "bf16" : "int8", kernels_name(), kernels_backend_name(), workers, k, reps, best, ops / best / 1e12,
         ops / best / 1e12 / workers);
  fflush(stdout);
  for (int w = 0; w < workers; w++)
    free(job.base[w]);
  free(job.base);
}

int main(int argc, char **argv) {
  int k = 512, reps = 20000, bf16 = 1, int8 = 1;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--kind") && i + 1 < argc) {
      i++;
      bf16 = strcmp(argv[i], "int8") != 0;
      int8 = strcmp(argv[i], "bf16") != 0;
    } else if (!strcmp(argv[i], "--k") && i + 1 < argc)
      k = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
      reps = atoi(argv[++i]);
    else {
      fprintf(stderr, "usage: amx-bench [--kind bf16|int8|both] [--k K] [--reps R]\n");
      return 2;
    }
  }
  kernels_init();
  if (!kernels_amx_active())
    fail("amx-bench needs LLAMA_KERNELS=amx");
  kernels_backend_init();
  if (bf16)
    bench(1, k, reps);
  if (int8)
    bench(0, k, reps);
  kernels_backend_free();
  return 0;
}
