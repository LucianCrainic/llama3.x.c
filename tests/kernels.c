/* Kernel regression: every available instruction set against double-precision
 * references, and bitwise agreement for RoPE and activation quantization. */
#include "kernels.h"
#include "workers.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long rng = 88172645463325252ull;
static int failures, checks;
static const char *current = "scalar";

static float uniform(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return (float)((double)(rng >> 40) / (double)(1ull << 24)) * 2.0f - 1.0f;
}
static void fill(float *x, size_t n, float scale) {
  for (size_t i = 0; i < n; i++)
    x[i] = uniform() * scale;
}
static void *zeroed(size_t count, size_t size) {
  void *p = calloc(count ? count : 1, size);
  assert(p);
  return p;
}
static void check(const char *what, double got, double expected, double tolerance) {
  checks++;
  if (!(fabs(got - expected) <= tolerance)) {
    if (failures++ < 20)
      fprintf(stderr, "%s [%s]: got %.9g expected %.9g (tolerance %.3g)\n", what, current, got, expected, tolerance);
  }
}
static double silu(double v) {
  return v / (1.0 + exp(-v));
}

static void test_f32(void) {
  const int shapes[][2] = {{4, 4}, {17, 5}, {64, 13}, {100, 3}, {1000, 9}, {4096, 6}, {48, 1}};
  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
    int n = shapes[s][0], d = shapes[s][1], dkv = d > 1 ? d / 2 : 1;
    float *x = zeroed(n, sizeof(float)), *w = zeroed((size_t)n * d, sizeof(float));
    float *w3 = zeroed((size_t)n * d, sizeof(float)), *wk = zeroed((size_t)n * dkv, sizeof(float));
    float *out = zeroed(d, sizeof(float)), *k = zeroed(dkv, sizeof(float)), *v = zeroed(dkv, sizeof(float));
    float *base = zeroed(d, sizeof(float));
    double *ref = zeroed(d, sizeof(double)), *mag = zeroed(d, sizeof(double));
    double *ref3 = zeroed(d, sizeof(double)), *mag3 = zeroed(d, sizeof(double));
    fill(x, n, 1);
    fill(w, (size_t)n * d, 1);
    fill(w3, (size_t)n * d, 1);
    fill(wk, (size_t)n * dkv, 1);
    fill(base, d, 4);
    for (int i = 0; i < d; i++)
      for (int j = 0; j < n; j++) {
        ref[i] += (double)w[(size_t)i * n + j] * x[j];
        mag[i] += fabs((double)w[(size_t)i * n + j] * x[j]);
        ref3[i] += (double)w3[(size_t)i * n + j] * x[j];
        mag3[i] += fabs((double)w3[(size_t)i * n + j] * x[j]);
      }
    k_matmul(out, x, w, n, d);
    for (int i = 0; i < d; i++)
      check("matmul", out[i], ref[i], 2e-5 * mag[i] + 1e-6);
    memcpy(out, base, d * sizeof(float));
    k_matmul_add(out, x, w, n, d);
    for (int i = 0; i < d; i++)
      check("matmul_add", out[i], base[i] + ref[i], 2e-5 * (mag[i] + fabs(base[i])) + 1e-6);
    k_matmul_swiglu(out, x, w, w3, n, d);
    for (int i = 0; i < d; i++) {
      double tolerance = 2e-5 * (mag[i] * (fabs(ref3[i]) + 1) + mag3[i] * (fabs(ref[i]) + 1)) + 1e-6;
      check("matmul_swiglu", out[i], silu(ref[i]) * ref3[i], tolerance);
    }
    /* QKV: q uses w, k uses wk, v uses the first dkv rows of w3. */
    k_matmul_qkv(out, k, v, x, w, wk, w3, n, d, dkv);
    for (int i = 0; i < d; i++)
      check("qkv.q", out[i], ref[i], 2e-5 * mag[i] + 1e-6);
    for (int i = 0; i < dkv; i++) {
      double rk = 0, mk = 0;
      for (int j = 0; j < n; j++) {
        rk += (double)wk[(size_t)i * n + j] * x[j];
        mk += fabs((double)wk[(size_t)i * n + j] * x[j]);
      }
      check("qkv.k", k[i], rk, 2e-5 * mk + 1e-6);
      check("qkv.v", v[i], ref3[i], 2e-5 * mag3[i] + 1e-6);
    }
    free(x), free(w), free(w3), free(wk), free(out), free(k), free(v), free(base);
    free(ref), free(mag), free(ref3), free(mag3);
  }
}

/* Returns the quantized activations so callers can compare instruction sets. */
static void test_q8(int8_t *quantized_out, float *scales_out) {
  const int shapes[][3] = {{64, 5, 16}, {192, 7, 32}, {256, 4, 64}, {4096, 9, 64}, {512, 3, 128}, {64, 2, 4}};
  size_t offset = 0, scale_offset = 0;
  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
    int n = shapes[s][0], d = shapes[s][1], gs = shapes[s][2], groups = n / gs;
    float *x = zeroed(n, sizeof(float)), *ws = zeroed((size_t)d * groups, sizeof(float));
    float *w3s = zeroed((size_t)d * groups, sizeof(float)), *xs = zeroed(groups, sizeof(float));
    int8_t *wq = zeroed((size_t)n * d, 1), *w3q = zeroed((size_t)n * d, 1), *xq = zeroed(n, 1);
    float *out = zeroed(d, sizeof(float)), *base = zeroed(d, sizeof(float));
    fill(x, n, 3);
    if (s == 0) {
      /* Group 0: scale exactly 1, so these hit roundf's half-away-from-zero ties. */
      const float ties[] = {127, 2.5f, -3.5f, 0.5f, -0.5f, 1.5f, -126.5f, 0};
      memcpy(x, ties, sizeof(ties));
      memset(x + 8, 0, 8 * sizeof(float));
      memset(x + 16, 0, 16 * sizeof(float)); /* an all-zero group */
    }
    for (size_t i = 0; i < (size_t)n * d; i++) {
      wq[i] = (int8_t)(uniform() * 127.0f);
      w3q[i] = (int8_t)(uniform() * 127.0f);
    }
    for (int i = 0; i < d * groups; i++) {
      ws[i] = 0.001f + 0.01f * fabsf(uniform());
      w3s[i] = 0.001f + 0.01f * fabsf(uniform());
    }
    fill(base, d, 2);
    k_quantize(xq, xs, x, n, gs);
    memcpy(quantized_out + offset, xq, n);
    memcpy(scales_out + scale_offset, xs, groups * sizeof(float));
    offset += n;
    scale_offset += groups;
    for (int i = 0; i < n; i++)
      check("quantize", xq[i] * (double)xs[i / gs], x[i], xs[i / gs] * 0.5 + 1e-7);
    for (int mode = 0; mode < 2; mode++) {
      K8Input in = mode ? (K8Input){NULL, NULL, x} : (K8Input){xq, xs, x};
      double *ref = zeroed(d, sizeof(double)), *mag = zeroed(d, sizeof(double));
      double *ref3 = zeroed(d, sizeof(double)), *mag3 = zeroed(d, sizeof(double));
      for (int i = 0; i < d; i++)
        for (int g = 0; g < groups; g++) {
          double a = 0, b = 0, am = 0, bm = 0;
          for (int j = g * gs; j < (g + 1) * gs; j++) {
            double xv = mode ? x[j] : xq[j];
            a += wq[(size_t)i * n + j] * xv;
            am += fabs(wq[(size_t)i * n + j] * xv);
            b += w3q[(size_t)i * n + j] * xv;
            bm += fabs(w3q[(size_t)i * n + j] * xv);
          }
          double sx = mode ? 1.0 : xs[g];
          ref[i] += a * ws[i * groups + g] * sx;
          mag[i] += am * ws[i * groups + g] * sx;
          ref3[i] += b * w3s[i * groups + g] * sx;
          mag3[i] += bm * w3s[i * groups + g] * sx;
        }
      const char *name = mode ? "q8_w8a32" : "q8_w8a8";
      K8Weight w = {wq, ws}, w3 = {w3q, w3s};
      k_matmul_q8(out, &in, w, n, d, gs);
      for (int i = 0; i < d; i++)
        check(name, out[i], ref[i], 2e-5 * mag[i] + 1e-6);
      memcpy(out, base, d * sizeof(float));
      k_matmul_q8_add(out, &in, w, n, d, gs);
      for (int i = 0; i < d; i++)
        check(name, out[i], base[i] + ref[i], 2e-5 * (mag[i] + fabs(base[i])) + 1e-6);
      k_matmul_q8_swiglu(out, &in, w, w3, n, d, gs);
      for (int i = 0; i < d; i++) {
        double tolerance = 2e-5 * (mag[i] * (fabs(ref3[i]) + 1) + mag3[i] * (fabs(ref[i]) + 1)) + 1e-6;
        check(name, out[i], silu(ref[i]) * ref3[i], tolerance);
      }
      float *q = zeroed(d, sizeof(float)), *k = zeroed(1, sizeof(float)), *v = zeroed(1, sizeof(float));
      k_matmul_q8_qkv(q, k, v, &in, w, w3, w, n, d, 1, gs);
      for (int i = 0; i < d; i++)
        check(name, q[i], ref[i], 2e-5 * mag[i] + 1e-6);
      check(name, k[0], ref3[0], 2e-5 * mag3[0] + 1e-6);
      check(name, v[0], ref[0], 2e-5 * mag[0] + 1e-6);
      free(q), free(k), free(v), free(ref), free(mag), free(ref3), free(mag3);
    }
    free(x), free(ws), free(w3s), free(xs), free(wq), free(w3q), free(xq), free(out), free(base);
  }
}

static void test_norm(void) {
  const int sizes[] = {4, 17, 64, 2048, 4099};
  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
    int n = sizes[s];
    float *x = zeroed(n, sizeof(float)), *w = zeroed(n, sizeof(float));
    float *got = zeroed(n, sizeof(float)), *ref = zeroed(n, sizeof(float));
    fill(x, n, 2);
    fill(w, n, 1);
    rmsnorm_eps(ref, x, w, n, 1e-5f);
    k_rmsnorm(got, x, w, n, 1e-5f);
    for (int i = 0; i < n; i++)
      check("rmsnorm", got[i], ref[i], 1e-5 * fabs(ref[i]) + 1e-7);
    k_rmsnorm(x, x, w, n, 1e-5f); /* in place, as for the final norm */
    for (int i = 0; i < n; i++)
      check("rmsnorm_inplace", x[i], ref[i], 1e-5 * fabs(ref[i]) + 1e-7);
    free(x), free(w), free(got), free(ref);
  }
}

/* Writes RoPE results for all head sizes so instruction sets can be compared. */
static size_t test_rope(float *result) {
  const int dims[] = {2, 16, 64, 128, 40};
  size_t used = 0;
  for (size_t s = 0; s < sizeof(dims) / sizeof(dims[0]); s++) {
    Config c = {0};
    c.head_dim = dims[s];
    c.n_heads = 6;
    c.n_kv_heads = 2;
    c.rope_theta = 500000.0f;
    c.rope_type = (int)(s % 2);
    c.rope_factor = 32;
    c.rope_low = 1;
    c.rope_high = 4;
    c.rope_original = 8192;
    size_t qn = (size_t)c.n_heads * c.head_dim, kn = (size_t)c.n_kv_heads * c.head_dim;
    float *table = zeroed(2 * (size_t)c.head_dim, sizeof(float));
    float *q = zeroed(qn, sizeof(float)), *k = zeroed(kn, sizeof(float));
    float *rq = zeroed(qn, sizeof(float)), *rk = zeroed(qn, sizeof(float));
    for (int position = 0; position < 300; position += 37) {
      fill(q, qn, 1);
      fill(k, kn, 1);
      memcpy(rq, q, qn * sizeof(float));
      memcpy(rk, k, kn * sizeof(float));
      apply_rope(rq, rk, &c, position);
      k_rope_table(&c, position, table);
      k_rope_apply(q, k, table, c.n_heads, c.n_kv_heads, c.head_dim);
      for (size_t i = 0; i < qn; i++)
        check("rope.q", q[i], rq[i], 1e-6);
      for (size_t i = 0; i < kn; i++)
        check("rope.k", k[i], rk[i], 1e-6);
      memcpy(result + used, q, qn * sizeof(float));
      used += qn;
      memcpy(result + used, k, kn * sizeof(float));
      used += kn;
    }
    free(table), free(q), free(k), free(rq), free(rk);
  }
  return used;
}

static void test_attention(void) {
  const int shapes[][4] = {{2, 1, 2, 0},
                           {2, 1, 2, 3},
                           {4, 2, 64, 0},
                           {4, 2, 64, 70},
                           {32, 8, 64, 300},
                           {32, 8, 128, 1000},
                           {24, 8, 128, 257},
                           {8, 8, 16, 40},
                           {32, 8, 128, 4095},
                           {6, 2, 40, 33},
                           /* chunk boundaries of attention_chunks: 32 and 256 positions */
                           {8, 2, 64, 31},
                           {8, 2, 64, 32},
                           {8, 2, 64, 255},
                           {8, 2, 64, 256}};
  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
    int heads = shapes[s][0], kv_heads = shapes[s][1], hd = shapes[s][2], pos = shapes[s][3];
    int seq = pos + 4, kv_dim = kv_heads * hd, kv_mul = heads / kv_heads;
    float *q = zeroed((size_t)heads * hd, sizeof(float)), *out = zeroed((size_t)heads * hd, sizeof(float));
    float *kc = zeroed((size_t)seq * kv_dim, sizeof(float)), *vc = zeroed((size_t)seq * kv_dim, sizeof(float));
    float *att = zeroed((size_t)heads * seq, sizeof(float));
    double *scores = zeroed(pos + 1, sizeof(double));
    fill(q, (size_t)heads * hd, 2);
    fill(kc, (size_t)seq * kv_dim, 2);
    fill(vc, (size_t)seq * kv_dim, 1);
    AttentionWork work;
    k_attention_init(&work, heads, kv_heads, hd, seq);
    k_attention(out, q, kc, vc, att, &work, heads, kv_heads, hd, seq, pos);
    for (int h = 0; h < heads; h++) {
      const float *kh = kc + (h / kv_mul) * hd, *vh = vc + (h / kv_mul) * hd;
      double max = -INFINITY, sum = 0;
      for (int t = 0; t <= pos; t++) {
        double dot = 0;
        for (int i = 0; i < hd; i++)
          dot += (double)q[h * hd + i] * kh[(size_t)t * kv_dim + i];
        scores[t] = dot / sqrt(hd);
        if (scores[t] > max)
          max = scores[t];
      }
      for (int t = 0; t <= pos; t++)
        sum += scores[t] = exp(scores[t] - max);
      for (int i = 0; i < hd; i++) {
        double ref = 0;
        for (int t = 0; t <= pos; t++)
          ref += scores[t] / sum * vh[(size_t)t * kv_dim + i];
        check("attention", out[h * hd + i], ref, 2e-5);
      }
    }
    k_attention_free(&work);
    free(q), free(out), free(kc), free(vc), free(att), free(scores);
  }
}

int main(void) {
  enum { ROPE_FLOATS = 1 << 16, Q8_VALUES = 1 << 14 };
  float *rope[2] = {zeroed(ROPE_FLOATS, sizeof(float)), zeroed(ROPE_FLOATS, sizeof(float))};
  int8_t *quantized[2] = {zeroed(Q8_VALUES, 1), zeroed(Q8_VALUES, 1)};
  float *scales[2] = {zeroed(Q8_VALUES, sizeof(float)), zeroed(Q8_VALUES, sizeof(float))};
  size_t rope_used = 0;
  int ran[2] = {0, 0};
  const KernelIsa isas[2] = {KERNELS_SCALAR, KERNELS_AVX512};
  for (int i = 0; i < 2; i++) {
    unsigned long long seed = rng;
    if (!kernels_select(isas[i])) {
      printf("kernels: %s unavailable on this CPU, skipped\n", i ? "avx512" : "scalar");
      continue;
    }
    current = kernels_name();
    int before = failures;
    test_f32();
    test_q8(quantized[i], scales[i]);
    test_norm();
    rope_used = test_rope(rope[i]);
    test_attention();
    rng = seed; /* identical inputs for every instruction set */
    ran[i] = 1;
    printf("kernels: %s %s\n", current, failures == before ? "ok" : "FAILED");
  }
  if (ran[0] && ran[1]) {
    current = "scalar vs avx512";
    checks += 3;
    if (memcmp(rope[0], rope[1], rope_used * sizeof(float)) && failures++ < 20)
      fprintf(stderr, "rope [%s]: results differ bitwise\n", current);
    if (memcmp(quantized[0], quantized[1], Q8_VALUES) && failures++ < 20)
      fprintf(stderr, "quantize [%s]: int8 values differ bitwise\n", current);
    if (memcmp(scales[0], scales[1], Q8_VALUES * sizeof(float)) && failures++ < 20)
      fprintf(stderr, "quantize [%s]: scales differ bitwise\n", current);
  }
  /* The same kernels on persistent worker teams. Odd sizes stress the range splitting, and every size runs
   * against the same double-precision references as the OpenMP path. */
  const int team_sizes[] = {1, 2, 3, 5, 7};
  int8_t *scratch_q = zeroed(Q8_VALUES, 1);
  float *scratch_s = zeroed(Q8_VALUES, sizeof(float));
  static char label[64];
  for (int i = 0; i < 2; i++) {
    if (!ran[i])
      continue;
    kernels_select(isas[i]);
    for (size_t t = 0; t < sizeof(team_sizes) / sizeof(*team_sizes); t++) {
      PoolConfig config = {.teams = 1, .threads = {team_sizes[t], 0}, .cpus = {NULL, NULL}};
      pool_start(&config);
      kernels_set_team(pool_team(0));
      snprintf(label, sizeof(label), "%s teams:%d", kernels_name(), team_sizes[t]);
      current = label;
      int before = failures;
      test_f32();
      test_q8(scratch_q, scratch_s);
      test_norm();
      test_attention();
      printf("kernels: %s %s\n", current, failures == before ? "ok" : "FAILED");
      kernels_set_team(NULL);
      pool_stop();
    }
  }
  free(scratch_q), free(scratch_s);
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
  free(rope[0]), free(rope[1]), free(quantized[0]), free(quantized[1]), free(scales[0]), free(scales[1]);
  printf("kernels: %d checks, %d failures\n", checks, failures);
  return failures != 0;
}
