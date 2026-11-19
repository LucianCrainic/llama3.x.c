/* GEMM microbenchmark (C2, extended for E6): the batched GEMM against the per-row GEMV on layer shapes, with the weights
 * in a region bound to one NUMA node. One JSON object per line on stdout.
 *   gemm-bench [--kernel f32|bf16|q8] [--node N] [--m 1,8,16,...] [--shapes NxK,...] [--reps R] [--gemv]
 *              [--cold-mib N] [--gs G]
 * --kernel f32 (default) is the AVX-512 FP32 GEMM; bf16 is the BF16 GEMM (the AMX tile kernel with LLAMA_KERNELS=amx,
 * the portable reference otherwise); q8 is the W8A8 Q8_0 GEMM (AVX-512 VNNI, or AMX-INT8 with LLAMA_KERNELS=amx, which
 * adds the packed int8 copy the engine keeps). --gemv times m calls of the per-row GEMV instead (f32: k_matmul, q8:
 * k_matmul_q8) as the per-token baseline; bf16 has none. --cold-mib N cycles through enough copies of the weights
 * (about N MiB in all, at least one) that each repetition reads memory the previous one left out of the caches; the
 * default 0 keeps the single copy of C2. q8 also reports the time of quantizing the m activation rows (quant_median_s),
 * which the engine pays once per linear and which is not part of best_s / median_s. Weights and activations are random.
 * Threads, pinning and the instruction set come from the engine's own environment: LLAMA_BACKEND=teams
 * LLAMA_THREADS=56 (under numactl --cpunodebind=0), LLAMA_KERNELS, LLAMA_GEMM_KC. */
#include "gemm.h"
#include "kernels.h"
#include "numa.h"
#include "workers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_REPS = 4096, MAX_COPIES = 256 };

static unsigned long long rng = 88172645463325252ull;
static float uniform(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return (float)((double)(rng >> 40) / (double)(1ull << 24)) * 2.0f - 1.0f;
}

static int parse_list(const char *text, int *out, int max) {
  int n = 0;
  char *copy = strdup(text);
  for (char *token = strtok(copy, ","); token && n < max; token = strtok(NULL, ","))
    out[n++] = atoi(token);
  free(copy);
  return n;
}

static int compare_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static size_t round_up(size_t v, size_t to) {
  return (v + to - 1) / to * to;
}

typedef enum { KERNEL_F32, KERNEL_BF16, KERNEL_Q8 } Kernel;

/* One copy of a matrix in whatever formats the kernel reads, all offsets inside the copy. */
typedef struct {
  size_t bytes, values, scales, packed; /* offsets; packed only for bf16 and for q8 with AMX */
  size_t streamed;                      /* bytes one GEMM call reads from the weights */
} Layout;

static Layout layout_of(Kernel kernel, int n, int k, int gs, int amx) {
  Layout l = {0};
  size_t cursor = 0;
  if (kernel == KERNEL_F32) {
    cursor = (size_t)n * k * sizeof(float);
    l.streamed = cursor;
  } else if (kernel == KERNEL_BF16) {
    cursor = bf16_vnni_elements(n, k) * sizeof(uint16_t);
    l.streamed = cursor;
  } else {
    size_t values = (size_t)n * k, scales = (size_t)n * (size_t)(k / gs) * sizeof(float);
    cursor = round_up(values, 64);
    l.scales = cursor, cursor += round_up(scales, 64);
    if (amx)
      l.packed = cursor, cursor += round_up(s8_vnni_bytes(n, k), 64);
    l.streamed = (amx ? s8_vnni_bytes(n, k) : values) + scales;
  }
  l.bytes = round_up(cursor, 64);
  return l;
}

static void fill_copy(Kernel kernel, char *copy, const Layout *l, int n, int k, int gs, int amx) {
  if (kernel == KERNEL_F32) {
    float *w = (float *)copy;
    for (size_t i = 0; i < (size_t)n * k; i++)
      w[i] = uniform() * 0.05f;
  } else if (kernel == KERNEL_BF16) {
    uint16_t *w = (uint16_t *)copy; /* any bf16 values are valid operands of a throughput test */
    for (size_t i = 0; i < bf16_vnni_elements(n, k); i++)
      w[i] = bf16_from_f32(uniform() * 0.05f);
  } else {
    int8_t *q = (int8_t *)copy;
    float *s = (float *)(copy + l->scales);
    for (size_t i = 0; i < (size_t)n * k; i++)
      q[i] = (int8_t)(uniform() * 120.0f);
    for (size_t i = 0; i < (size_t)n * (size_t)(k / gs); i++)
      s[i] = 0.001f + 0.0005f * (uniform() + 1.0f);
    if (amx)
      pack_s8_vnni((int8_t *)(copy + l->packed), q, n, k, (size_t)k);
  }
}

int main(int argc, char **argv) {
  int node = -1, reps = 20, gemv = 0, gs = 64, cold_mib = 0, ms[32] = {1, 8, 16, 32, 64, 128, 256}, nm = 7;
  int shapes[16][2] = {{4096, 4096}, {1024, 4096}, {14336, 4096}, {4096, 14336}}, nshapes = 4;
  Kernel kernel = KERNEL_F32;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--node") && i + 1 < argc)
      node = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
      reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--gs") && i + 1 < argc)
      gs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--cold-mib") && i + 1 < argc)
      cold_mib = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--kernel") && i + 1 < argc) {
      i++;
      if (strcmp(argv[i], "f32") && strcmp(argv[i], "bf16") && strcmp(argv[i], "q8")) {
        fprintf(stderr, "gemm-bench: --kernel must be f32, bf16 or q8\n");
        return 2;
      }
      kernel = !strcmp(argv[i], "bf16") ? KERNEL_BF16 : !strcmp(argv[i], "q8") ? KERNEL_Q8 : KERNEL_F32;
    } else if (!strcmp(argv[i], "--m") && i + 1 < argc)
      nm = parse_list(argv[++i], ms, 32);
    else if (!strcmp(argv[i], "--gemv"))
      gemv = 1;
    else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) {
      nshapes = 0;
      char *copy = strdup(argv[++i]);
      for (char *token = strtok(copy, ","); token && nshapes < 16; token = strtok(NULL, ","))
        if (sscanf(token, "%dx%d", &shapes[nshapes][0], &shapes[nshapes][1]) == 2)
          nshapes++;
      free(copy);
    } else {
      fprintf(stderr, "usage: gemm-bench [--kernel f32|bf16|q8] [--node N] [--m LIST] [--shapes NxK,...] [--reps R] "
                      "[--gemv] [--cold-mib N] [--gs G]\n");
      return 2;
    }
  }
  if (reps < 1 || reps > MAX_REPS || gs < 1 || cold_mib < 0 || (gemv && kernel == KERNEL_BF16)) {
    fprintf(stderr, "gemm-bench: bad --reps, --gs or --cold-mib, or --gemv with --kernel bf16\n");
    return 2;
  }
  kernels_init();
  kernels_backend_init();
  int amx = gemm_amx_active();
  int max_m = 0;
  for (int i = 0; i < nm; i++)
    max_m = ms[i] > max_m ? ms[i] : max_m;
  const char *kname[2][3] = {{"gemm", "gemm_bf16", "gemm_q8"}, {"gemv_rows", "-", "gemv_rows_q8"}};
  for (int s = 0; s < nshapes; s++) {
    int n = shapes[s][0], k = shapes[s][1];
    if (kernel == KERNEL_Q8 && k % gs) {
      fprintf(stderr, "gemm-bench: K=%d is not a multiple of the group size %d\n", k, gs);
      return 2;
    }
    Layout layout = layout_of(kernel, n, k, gs, amx);
    int copies = cold_mib ? (int)(((double)cold_mib * (1 << 20)) / (double)layout.bytes) : 1;
    copies = copies < 1 ? 1 : copies > MAX_COPIES ? MAX_COPIES : copies;
    NumaRegion region;
    region_reserve(&region, layout.bytes * (size_t)copies, 0, node, ROLE_WEIGHTS, 0);
    char *base = region.data;
    fill_copy(kernel, base, &layout, n, k, gs, amx);
    for (int c = 1; c < copies; c++)
      memcpy(base + (size_t)c * layout.bytes, base, layout.bytes);
    region_mark_populated(&region);
    size_t rows_alloc = round_up((size_t)max_m, 32) + 32;
    float *a = calloc(rows_alloc * (size_t)k, sizeof(float)), *c = calloc((size_t)max_m * n, sizeof(float));
    for (size_t i = 0; i < (size_t)max_m * k; i++)
      a[i] = uniform();
    int8_t *xq = NULL;
    float *xs = NULL;
    AlignedBuffer panel = {0};
    if (kernel == KERNEL_Q8) {
      xq = calloc(rows_alloc * (size_t)k, 1);
      xs = calloc(rows_alloc * (size_t)(k / gs), sizeof(float));
    } else if (kernel == KERNEL_BF16)
      aligned_buffer_alloc(&panel, k_gemm_bf16_panel_bytes(max_m, k), MEM_ACTIVATION);
    for (int mi = 0; mi < nm; mi++) {
      int m = ms[mi];
      if (kernel == KERNEL_Q8)
        k_quantize_rows(xq, xs, a, (size_t)k, m, k, gs);
      for (int mode = 0; mode < (gemv ? 2 : 1); mode++) {
        double times[MAX_REPS];
        for (int pass = -2; pass < reps; pass++) { /* two warm-up calls, then the timed ones */
          const char *w = base + (size_t)((pass < 0 ? pass + 2 : pass) % copies) * layout.bytes;
          double start = monotonic_seconds();
          if (kernel == KERNEL_F32) {
            const float *wf = (const float *)w;
            GemmSegment seg = {.c = c, .ldc = n, .w = wf, .ldw = (size_t)k, .n = n, .kind = GEMM_STORE};
            if (mode)
              for (int i = 0; i < m; i++)
                k_matmul(c + (size_t)i * n, a + (size_t)i * k, wf, k, n);
            else
              k_gemm_f32(a, k, m, k, &seg, 1);
          } else if (kernel == KERNEL_BF16) {
            GemmBf16Segment seg = {.c = c, .ldc = n, .w = (const uint16_t *)w, .n = n, .kind = GEMM_STORE};
            k_gemm_bf16(a, k, m, k, &seg, 1, &panel);
          } else {
            const int8_t *values = (const int8_t *)w, *packed = (const int8_t *)(w + layout.packed);
            const float *scales = (const float *)(w + layout.scales);
            if (mode) {
              for (int i = 0; i < m; i++) {
                K8Input in = {.q = xq + (size_t)i * k, .s = xs + (size_t)i * (k / gs), .f = NULL};
                k_matmul_q8(c + (size_t)i * n, &in, (K8Weight){.q = values, .s = scales}, k, n, gs);
              }
            } else {
              /* With AMX the kernel reads the packed copy; the unpacked pointer is still the matrix's own. */
              GemmQ8Segment seg = {.c = c, .ldc = n, .w = {.q = values, .s = scales}, .pw = amx ? packed : NULL, .n = n,
                                   .kind = GEMM_STORE};
              k_gemm_q8(xq, xs, m, k, gs, &seg, 1);
            }
          }
          double elapsed = monotonic_seconds() - start;
          if (pass >= 0)
            times[pass] = elapsed;
        }
        double sorted[MAX_REPS], total = 0;
        memcpy(sorted, times, (size_t)reps * sizeof(double));
        qsort(sorted, (size_t)reps, sizeof(double), compare_double);
        for (int r = 0; r < reps; r++)
          total += times[r];
        /* A call below the clock's resolution reads as 1 ns, so the rates below stay finite (valid JSON) on toy shapes. */
        double best = sorted[0] > 1e-9 ? sorted[0] : 1e-9, median = sorted[reps / 2] > 1e-9 ? sorted[reps / 2] : 1e-9;
        double mean = total / reps > 1e-9 ? total / reps : 1e-9;
        double quant = 0;
        if (kernel == KERNEL_Q8 && !mode) {
          double qt[MAX_REPS];
          for (int r = 0; r < reps; r++) {
            double start = monotonic_seconds();
            k_quantize_rows(xq, xs, a, (size_t)k, m, k, gs);
            qt[r] = monotonic_seconds() - start;
          }
          qsort(qt, (size_t)reps, sizeof(double), compare_double);
          quant = qt[reps / 2];
        }
        double flops = 2.0 * m * n * k;
        /* Weight bytes are read once per call by the GEMM and once per row by the GEMV. The GEMV reads the unpacked
         * matrix, so for q8 with AMX its bytes are the int8 values plus scales, not the packed copy. */
        double per_pass = (double)layout.streamed;
        if (kernel == KERNEL_Q8 && mode && amx)
          per_pass = (double)n * k + (double)n * (k / gs) * sizeof(float);
        double streamed = per_pass * (mode ? m : 1);
        const char *impl = kernel == KERNEL_Q8
                               ? (mode ? "gemv_q8" : amx ? "amx_int8" : kernels_avx512_active() ? "avx512_q8" : "portable")
                           : kernel == KERNEL_BF16 ? (amx ? "amx_bf16" : "portable_bf16")
                                                   : "f32";
        printf("{\"kernel\": \"%s\", \"impl\": \"%s\", \"isa\": \"%s\", \"backend\": \"%s\", \"node\": %d, \"n\": %d, "
               "\"k\": %d, \"m\": %d, \"reps\": %d, \"cold_copies\": %d, \"weight_bytes\": %.0f, \"best_s\": %.9f, "
               "\"median_s\": %.9f, \"mean_s\": %.9f, \"gflops_best\": %.2f, \"gflops_median\": %.2f, "
               "\"gflops_mean\": %.2f, \"weight_gbs_best\": %.2f, \"weight_gbs_median\": %.2f, "
               "\"weight_gbs_mean\": %.2f, \"quant_median_s\": %.9f, \"kc\": %d}\n",
               kname[mode][kernel], impl, kernels_name(), kernels_backend_name(), region.node, n, k, m, reps, copies,
               per_pass, best, median, mean, flops / best / 1e9, flops / median / 1e9, flops / mean / 1e9,
               streamed / best / 1e9, streamed / median / 1e9, streamed / mean / 1e9, quant, k_gemm_kc());
        fflush(stdout);
      }
    }
    free(a), free(c), free(xq), free(xs);
    if (panel.base)
      aligned_buffer_free(&panel);
    region_release(&region);
  }
  kernels_backend_free();
  return 0;
}
