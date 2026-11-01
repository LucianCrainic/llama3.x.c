/* GEMM microbenchmark (C2): the batched FP32 GEMM against the per-row GEMV on the 8B layer shapes, with the weights
 * in a region bound to one NUMA node. One JSON object per line on stdout.
 *   gemm-bench [--node N] [--m 1,8,16,...] [--shapes NxK,...] [--reps R] [--gemv]
 * Threads, pinning and the instruction set come from the engine's own environment: LLAMA_BACKEND=teams
 * LLAMA_THREADS=56 (under numactl --cpunodebind=0), LLAMA_KERNELS, LLAMA_GEMM_KC. */
#include "gemm.h"
#include "kernels.h"
#include "numa.h"
#include "workers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long rng = 88172645463325252ull;
static float uniform(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return (float)((double)(rng >> 40) / (double)(1ull << 24)) * 2.0f - 1.0f;
}

static int parse_list(const char *text, int *out, int max) {
  int n = 0;
  for (char *copy = strdup(text), *token = strtok(copy, ","); token && n < max; token = strtok(NULL, ","))
    out[n++] = atoi(token);
  return n;
}

int main(int argc, char **argv) {
  int node = -1, reps = 20, gemv = 0, ms[32] = {1, 8, 16, 32, 64, 128, 256}, nm = 7;
  int shapes[16][2] = {{4096, 4096}, {1024, 4096}, {14336, 4096}, {4096, 14336}}, nshapes = 4;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--node") && i + 1 < argc)
      node = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
      reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--m") && i + 1 < argc)
      nm = parse_list(argv[++i], ms, 32);
    else if (!strcmp(argv[i], "--gemv"))
      gemv = 1;
    else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) {
      nshapes = 0;
      for (char *copy = strdup(argv[++i]), *token = strtok(copy, ","); token && nshapes < 16; token = strtok(NULL, ","))
        if (sscanf(token, "%dx%d", &shapes[nshapes][0], &shapes[nshapes][1]) == 2)
          nshapes++;
    } else {
      fprintf(stderr, "usage: gemm-bench [--node N] [--m LIST] [--shapes NxK,...] [--reps R] [--gemv]\n");
      return 2;
    }
  }
  kernels_init();
  kernels_backend_init();
  int max_m = 0;
  for (int i = 0; i < nm; i++)
    max_m = ms[i] > max_m ? ms[i] : max_m;
  for (int s = 0; s < nshapes; s++) {
    int n = shapes[s][0], k = shapes[s][1];
    size_t bytes = (size_t)n * k * sizeof(float);
    NumaRegion region;
    region_reserve(&region, bytes, 0, node, ROLE_WEIGHTS, 0);
    float *w = region.data;
    for (size_t i = 0; i < (size_t)n * k; i++)
      w[i] = uniform() * 0.05f;
    region_mark_populated(&region);
    float *a = calloc((size_t)max_m * k, sizeof(float)), *c = calloc((size_t)max_m * n, sizeof(float));
    for (size_t i = 0; i < (size_t)max_m * k; i++)
      a[i] = uniform();
    for (int mi = 0; mi < nm; mi++) {
      int m = ms[mi];
      for (int mode = 0; mode < (gemv ? 2 : 1); mode++) {
        GemmSegment seg = {.c = c, .ldc = n, .w = w, .ldw = (size_t)k, .n = n, .kind = GEMM_STORE};
        for (int warm = 0; warm < 2; warm++) {
          if (mode)
            for (int i = 0; i < m; i++)
              k_matmul(c + (size_t)i * n, a + (size_t)i * k, w, k, n);
          else
            k_gemm_f32(a, k, m, k, &seg, 1);
        }
        double best = 1e30, total = 0;
        for (int r = 0; r < reps; r++) {
          double start = monotonic_seconds();
          if (mode)
            for (int i = 0; i < m; i++)
              k_matmul(c + (size_t)i * n, a + (size_t)i * k, w, k, n);
          else
            k_gemm_f32(a, k, m, k, &seg, 1);
          double elapsed = monotonic_seconds() - start;
          total += elapsed;
          best = elapsed < best ? elapsed : best;
        }
        double flops = 2.0 * m * n * k, mean = total / reps;
        /* Weight bytes are read once per call by the GEMM and once per row by the GEMV. */
        double streamed = (double)bytes * (mode ? m : 1);
        printf("{\"kernel\": \"%s\", \"isa\": \"%s\", \"backend\": \"%s\", \"node\": %d, \"n\": %d, \"k\": %d, \"m\": %d, "
               "\"reps\": %d, \"best_s\": %.9f, \"mean_s\": %.9f, \"gflops_best\": %.2f, \"gflops_mean\": %.2f, "
               "\"weight_gbs_best\": %.2f, \"weight_gbs_mean\": %.2f, \"kc\": %d}\n",
               mode ? "gemv_rows" : "gemm", kernels_name(), kernels_backend_name(), region.node, n, k, m, reps, best,
               mean, flops / best / 1e9, flops / mean / 1e9, streamed / best / 1e9, streamed / mean / 1e9,
               k_gemm_kc());
        fflush(stdout);
      }
    }
    free(a), free(c);
    region_release(&region);
  }
  kernels_backend_free();
  return 0;
}
