/* Kernel regression: every available instruction set against double-precision
 * references, and bitwise agreement for RoPE and activation quantization. */
#include "kernels.h"
#include "gemm.h"
#include "numa.h"
#include "testing.h"
#include "workers.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
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
  /* gs 32 with n % 64 == 0 takes the VNNI pair kernel; {96, 5, 32} does not (n % 64), and stays portable. */
  const int shapes[][3] = {{64, 5, 16},   {192, 7, 32}, {256, 4, 64},  {4096, 9, 64},
                           {512, 3, 128}, {64, 2, 4},   {4096, 9, 32}, {96, 5, 32}};
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

/* C3: batched causal attention. Row i attends to 0..pos0+i and equals the per-row call; K/V beyond a row never matter. */
static void test_attention_batch(void) {
  const int configs[][3] = {{4, 4, 64}, {8, 2, 64}, {8, 2, 128}, {6, 2, 40}};
  const int ms[] = {1, 2, 7, 32}, starts[] = {0, 31, 250, 4000};
  for (size_t c = 0; c < sizeof(configs) / sizeof(*configs); c++)
    for (size_t mi = 0; mi < sizeof(ms) / sizeof(*ms); mi++)
      for (size_t pi = 0; pi < sizeof(starts) / sizeof(*starts); pi++) {
        int heads = configs[c][0], kv_heads = configs[c][1], hd = configs[c][2], m = ms[mi], pos0 = starts[pi];
        if (pos0 == 4000 && (m == 2 || hd == 40 || kv_heads == heads))
          continue; /* keep the long-context cases to a few shapes */
        int seq = pos0 + m; /* the last row sits exactly on the context limit */
        size_t dim = (size_t)heads * hd, kv_dim = (size_t)kv_heads * hd;
        float *q = zeroed(m * dim, sizeof(float)), *kc = zeroed(seq * kv_dim, sizeof(float));
        float *vc = zeroed(seq * kv_dim, sizeof(float)), *att = zeroed((size_t)heads * seq, sizeof(float));
        float *want = zeroed(m * dim, sizeof(float)), *got = zeroed(m * dim, sizeof(float));
        float *again = zeroed(m * dim, sizeof(float)), *port = zeroed(m * dim, sizeof(float));
        fill(q, m * dim, 2), fill(kc, seq * kv_dim, 2), fill(vc, seq * kv_dim, 1);
        AttentionWork work;
        k_attention_init_rows(&work, heads, kv_heads, hd, seq, m, NULL);
        for (int i = 0; i < m; i++)
          k_attention(want + i * dim, q + i * dim, kc, vc, att, &work, heads, kv_heads, hd, seq, pos0 + i);
        size_t full_capacity = work.capacity, one_row = (size_t)heads * work.max_chunks * (hd + 2);
        for (int pass = 0; pass < 2; pass++) {
          /* pass 1: room for only two rows, so a long batch runs in several waves */
          work.capacity = pass ? (2 * one_row < full_capacity ? 2 * one_row : full_capacity) : full_capacity;
          memset(got, 0xAB, m * dim * sizeof(float)); /* poison: every row must be written by this call */
          k_attention_batch(got, q, kc, vc, att, &work, heads, kv_heads, hd, seq, pos0, m);
          checks++;
          if (memcmp(got, want, m * dim * sizeof(float)) && failures++ < 20)
            fprintf(stderr, "attention_batch [%s] heads=%d kv=%d hd=%d m=%d pos0=%d pass=%d: differs from per-row\n",
                    current, heads, kv_heads, hd, m, pos0, pass);
          if (m > 1) {
            memset(port, 0xAB, m * dim * sizeof(float));
            k_attention_batch_portable(port, q, kc, vc, &work, heads, kv_heads, hd, seq, pos0, m);
            for (size_t i = 0; i < m * dim; i++)
              check("attention_batch.portable", port[i], want[i], 2e-5);
          }
        }
        work.capacity = full_capacity;
        /* Causality: change every K/V row after the last position row i may see; row i must not move, bit for bit. */
        const int probes[] = {0, m / 2, m - 1};
        for (size_t pr = 0; pr < 3; pr++) {
          int i = probes[pr], first = pos0 + i + 1;
          if (first >= seq)
            continue;
          float *kp = zeroed(seq * kv_dim, sizeof(float)), *vp = zeroed(seq * kv_dim, sizeof(float));
          memcpy(kp, kc, seq * kv_dim * sizeof(float)), memcpy(vp, vc, seq * kv_dim * sizeof(float));
          for (size_t t = (size_t)first * kv_dim; t < seq * kv_dim; t++)
            kp[t] += 3 * uniform(), vp[t] += 3 * uniform();
          memset(again, 0xAB, m * dim * sizeof(float));
          k_attention_batch(again, q, kp, vp, att, &work, heads, kv_heads, hd, seq, pos0, m);
          checks++;
          if (memcmp(again + i * dim, got + i * dim, dim * sizeof(float)) && failures++ < 20)
            fprintf(stderr, "attention_batch causality [%s] m=%d pos0=%d row=%d: later K/V changed the row\n", current,
                    m, pos0, i);
          if (m > 1) {
            float *port2 = zeroed(m * dim, sizeof(float));
            memset(port2, 0xAB, m * dim * sizeof(float));
            k_attention_batch_portable(port2, q, kp, vp, &work, heads, kv_heads, hd, seq, pos0, m);
            checks++;
            if (memcmp(port2 + i * dim, port + i * dim, dim * sizeof(float)) && failures++ < 20)
              fprintf(stderr, "attention_batch portable causality [%s] m=%d pos0=%d row=%d\n", current, m, pos0, i);
            free(port2);
          }
          free(kp), free(vp);
        }
        k_attention_free(&work);
        free(q), free(kc), free(vc), free(att), free(want), free(got), free(again), free(port);
      }
}

/* Waves: with room for only two rows of partials the batch runs in several waves whose row blocks are cut short; every
 * row is computed as before, so the result is bitwise the one-wave result, on the engine path and the portable driver.
 * (E0 also checked every row block 1..16; the block is fixed at 4 since F5.) */
static void test_attention_batch_waves(void) {
  const int configs[][3] = {{8, 2, 64}, {6, 2, 40}, {4, 4, 128}};
  const int ms[] = {2, 7, 13, 32}, starts[] = {0, 31, 250};
  for (size_t c = 0; c < sizeof(configs) / sizeof(*configs); c++)
    for (size_t mi = 0; mi < sizeof(ms) / sizeof(*ms); mi++)
      for (size_t pi = 0; pi < sizeof(starts) / sizeof(*starts); pi++) {
        int heads = configs[c][0], kv_heads = configs[c][1], hd = configs[c][2], m = ms[mi], pos0 = starts[pi];
        int seq = pos0 + m;
        size_t dim = (size_t)heads * hd, kv_dim = (size_t)kv_heads * hd;
        float *q = zeroed(m * dim, sizeof(float)), *kc = zeroed(seq * kv_dim, sizeof(float));
        float *vc = zeroed(seq * kv_dim, sizeof(float)), *att = zeroed((size_t)heads * seq, sizeof(float));
        float *want = zeroed(m * dim, sizeof(float)), *got = zeroed(m * dim, sizeof(float));
        float *want_port = zeroed(m * dim, sizeof(float)), *port = zeroed(m * dim, sizeof(float));
        fill(q, m * dim, 2), fill(kc, seq * kv_dim, 2), fill(vc, seq * kv_dim, 1);
        AttentionWork work;
        k_attention_init_rows(&work, heads, kv_heads, hd, seq, m, NULL);
        size_t full_capacity = work.capacity, one_row = (size_t)heads * work.max_chunks * (hd + 2);
        k_attention_batch(want, q, kc, vc, att, &work, heads, kv_heads, hd, seq, pos0, m);
        k_attention_batch_portable(want_port, q, kc, vc, &work, heads, kv_heads, hd, seq, pos0, m);
        work.capacity = 2 * one_row < full_capacity ? 2 * one_row : full_capacity;
        memset(got, 0xAB, m * dim * sizeof(float));
        memset(port, 0xAB, m * dim * sizeof(float));
        k_attention_batch(got, q, kc, vc, att, &work, heads, kv_heads, hd, seq, pos0, m);
        k_attention_batch_portable(port, q, kc, vc, &work, heads, kv_heads, hd, seq, pos0, m);
        checks += 2;
        if (memcmp(got, want, m * dim * sizeof(float)) && failures++ < 20)
          fprintf(stderr, "attention_batch waves [%s] heads=%d kv=%d hd=%d m=%d pos0=%d: differs\n", current, heads,
                  kv_heads, hd, m, pos0);
        if (memcmp(port, want_port, m * dim * sizeof(float)) && failures++ < 20)
          fprintf(stderr, "attention_batch portable waves [%s] heads=%d m=%d pos0=%d: differs\n", current, heads, m, pos0);
        work.capacity = full_capacity;
        k_attention_free(&work);
        free(q), free(kc), free(vc), free(att), free(want), free(got), free(want_port), free(port);
      }
}

/* ---- Shard kernels (B2): partial outputs, column slices, row slices and padded rows -------------------------------
 */

static int bitwise_expected(void) {
  /* The AVX-512 GEMV reduces a block of four rows differently from one to three rows, so a row computed in another
   * block position can differ in the last bits. Only the scalar kernels keep the exact per-row order. */
  return !strcmp(kernels_name(), "scalar");
}

static void test_partial_f32(void) {
  const int shapes[][2] = {{64, 5}, {96, 9}, {200, 4}, {1024, 7}};
  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
    int n = shapes[s][0], d = shapes[s][1], half = n / 2;
    for (int pad = 0; pad < 2; pad++) {
      size_t ld = (size_t)half + (pad ? 16 : 0);
      float *x = zeroed(n, sizeof(float)), *w = zeroed((size_t)n * d, sizeof(float));
      float *w0 = zeroed(ld * d, sizeof(float)), *w1 = zeroed(ld * d, sizeof(float));
      float *p0 = zeroed(d, sizeof(float)), *p1 = zeroed(d, sizeof(float)), *full = zeroed(d, sizeof(float));
      float *base = zeroed(d, sizeof(float));
      double *ref = zeroed(d, sizeof(double)), *mag = zeroed(d, sizeof(double));
      fill(x, n, 1);
      fill(w, (size_t)n * d, 1);
      fill(base, d, 4);
      for (size_t i = 0; i < ld * d; i++)
        w0[i] = w1[i] = NAN; /* padding must never be read */
      for (int i = 0; i < d; i++) {
        memcpy(w0 + i * ld, w + (size_t)i * n, half * sizeof(float));
        memcpy(w1 + i * ld, w + (size_t)i * n + half, (n - half) * sizeof(float));
        for (int j = 0; j < n; j++) {
          ref[i] += (double)w[(size_t)i * n + j] * x[j];
          mag[i] += fabs((double)w[(size_t)i * n + j] * x[j]);
        }
      }
      for (int i = 0; i < d; i++)
        p0[i] = p1[i] = 1e30f; /* PARTIAL overwrites, like STORE */
      k_matmul_ld(p0, x, w0, ld, half, d, K_PARTIAL);
      k_matmul_ld(p1, x + half, w1, ld, n - half, d, K_PARTIAL);
      for (int i = 0; i < d; i++)
        check("partial.sum", (double)p0[i] + (double)p1[i], ref[i], 2e-5 * mag[i] + 1e-6);
      /* Row slices of the full tensor: the shard's rows are the full product's rows. */
      k_matmul(full, x, w, n, d);
      int split = d / 2;
      memcpy(p0, base, d * sizeof(float));
      k_matmul_ld(p0, x, w, (size_t)n, n, split, K_STORE);
      k_matmul_ld(p0 + split, x, w + (size_t)split * n, (size_t)n, n, d - split, K_STORE);
      for (int i = 0; i < d; i++) {
        checks++;
        if (bitwise_expected() ? memcmp(&p0[i], &full[i], sizeof(float)) != 0
                               : fabs((double)p0[i] - full[i]) > 2e-5 * mag[i] + 1e-6) {
          if (failures++ < 20)
            fprintf(stderr, "row slice [%s]: row %d differs from the full product\n", current, i);
        }
      }
      /* K_ADD with padded rows accumulates; SWIGLU on padded rows matches the unpadded pair form. */
      memcpy(p1, base, d * sizeof(float));
      k_matmul_ld(p1, x, w0, ld, half, d, K_ADD);
      for (int i = 0; i < d; i++) {
        double ref0 = 0, mag0 = 0;
        for (int j = 0; j < half; j++) {
          ref0 += (double)w[(size_t)i * n + j] * x[j];
          mag0 += fabs((double)w[(size_t)i * n + j] * x[j]);
        }
        check("partial.add", p1[i], base[i] + ref0, 2e-5 * (mag0 + fabs(base[i])) + 1e-6);
      }
      float *gated = zeroed(d, sizeof(float)), *unpadded = zeroed(d, sizeof(float));
      float *w3 = zeroed(ld * d, sizeof(float)), *w3_plain = zeroed((size_t)half * d, sizeof(float));
      for (size_t i = 0; i < ld * d; i++)
        w3[i] = NAN;
      for (int i = 0; i < d; i++)
        for (int j = 0; j < half; j++)
          w3[i * ld + j] = w3_plain[(size_t)i * half + j] = uniform();
      k_matmul_swiglu_ld(gated, x, w0, w3, ld, half, d);
      float *w0_plain = zeroed((size_t)half * d, sizeof(float));
      for (int i = 0; i < d; i++)
        memcpy(w0_plain + (size_t)i * half, w0 + i * ld, half * sizeof(float));
      k_matmul_swiglu(unpadded, x, w0_plain, w3_plain, half, d);
      for (int i = 0; i < d; i++) {
        checks++;
        if (memcmp(&gated[i], &unpadded[i], sizeof(float)) && failures++ < 20)
          fprintf(stderr, "swiglu_ld [%s]: padded and packed rows differ at %d\n", current, i);
      }
      free(gated), free(unpadded), free(w3), free(w3_plain), free(w0_plain);
      free(x), free(w), free(w0), free(w1), free(p0), free(p1), free(full), free(base), free(ref), free(mag);
    }
  }
}

static void test_partial_q8(void) {
  const int gss[] = {4, 16, 32, 64, 128};
  const int n = 512, d = 7, half = n / 2;
  for (size_t g = 0; g < sizeof(gss) / sizeof(*gss); g++) {
    int gs = gss[g], groups = n / gs, hg = half / gs;
    for (int pad = 0; pad < 2; pad++) {
      size_t ld = (size_t)half + (pad ? (size_t)gs : 0), sld = ld / gs;
      float *x = zeroed(n, sizeof(float)), *ws = zeroed((size_t)d * groups, sizeof(float));
      int8_t *wq = zeroed((size_t)n * d, 1);
      int8_t *w0q = zeroed(ld * d, 1), *w1q = zeroed(ld * d, 1);
      float *w0s = zeroed(sld * d, sizeof(float)), *w1s = zeroed(sld * d, sizeof(float));
      fill(x, n, 3);
      for (size_t i = 0; i < (size_t)n * d; i++)
        wq[i] = (int8_t)(uniform() * 127.0f);
      for (int i = 0; i < d * groups; i++)
        ws[i] = 0.001f + 0.01f * fabsf(uniform());
      memset(w0q, 127, ld * d), memset(w1q, 127, ld * d); /* padding */
      for (size_t i = 0; i < sld * d; i++)
        w0s[i] = w1s[i] = NAN;
      for (int i = 0; i < d; i++) {
        memcpy(w0q + i * ld, wq + (size_t)i * n, half);
        memcpy(w1q + i * ld, wq + (size_t)i * n + half, half);
        memcpy(w0s + i * sld, ws + (size_t)i * groups, hg * sizeof(float));
        memcpy(w1s + i * sld, ws + (size_t)i * groups + hg, hg * sizeof(float));
      }
      int8_t *xq = zeroed(n, 1);
      float *xs = zeroed(groups, sizeof(float));
      k_quantize(xq, xs, x, n, gs);
      /* Each half's activations are quantized from the owned slice alone, at the original group boundaries. */
      int8_t *xq0 = zeroed(half, 1), *xq1 = zeroed(half, 1);
      float *xs0 = zeroed(hg, sizeof(float)), *xs1 = zeroed(hg, sizeof(float));
      k_quantize(xq0, xs0, x, half, gs);
      k_quantize(xq1, xs1, x + half, half, gs);
      checks += 2;
      if ((memcmp(xq0, xq, half) || memcmp(xq1, xq + half, half) || memcmp(xs0, xs, hg * sizeof(float)) ||
           memcmp(xs1, xs + hg, hg * sizeof(float))) &&
          failures++ < 20)
        fprintf(stderr, "quantize slices [%s]: differ from the full quantization (gs %d)\n", current, gs);
      for (int mode = 0; mode < 2; mode++) {
        double *ref = zeroed(d, sizeof(double)), *mag = zeroed(d, sizeof(double));
        for (int i = 0; i < d; i++)
          for (int gi = 0; gi < groups; gi++) {
            double a = 0, am = 0;
            for (int j = gi * gs; j < (gi + 1) * gs; j++) {
              double xv = mode ? x[j] : xq[j];
              a += wq[(size_t)i * n + j] * xv;
              am += fabs(wq[(size_t)i * n + j] * xv);
            }
            double sx = mode ? 1.0 : xs[gi];
            ref[i] += a * ws[i * groups + gi] * sx;
            mag[i] += am * ws[i * groups + gi] * sx;
          }
        K8Input in0 = mode ? (K8Input){NULL, NULL, x} : (K8Input){xq0, xs0, x};
        K8Input in1 = mode ? (K8Input){NULL, NULL, x + half} : (K8Input){xq1, xs1, x + half};
        K8Weight a = {w0q, w0s}, b = {w1q, w1s};
        float *p0 = zeroed(d, sizeof(float)), *p1 = zeroed(d, sizeof(float)), *both = zeroed(d, sizeof(float));
        for (int i = 0; i < d; i++)
          p0[i] = p1[i] = 1e30f;
        k_matmul_q8_ld(p0, &in0, &a, ld, half, d, gs, K_PARTIAL);
        k_matmul_q8_ld(p1, &in1, &b, ld, half, d, gs, K_PARTIAL);
        const char *name = mode ? "q8_partial_w8a32" : "q8_partial_w8a8";
        for (int i = 0; i < d; i++)
          check(name, (double)p0[i] + (double)p1[i], ref[i], 2e-5 * mag[i] + 1e-6);
        /* Padded rows and packed rows give the same bits, and ADD accumulates. */
        K8Input full_in = mode ? (K8Input){NULL, NULL, x} : (K8Input){xq, xs, x};
        K8Weight plain = {wq, ws};
        k_matmul_q8(both, &full_in, plain, n, d, gs);
        for (int i = 0; i < d; i++)
          check(name, both[i], ref[i], 2e-5 * mag[i] + 1e-6);
        float *add = zeroed(d, sizeof(float));
        for (int i = 0; i < d; i++)
          add[i] = 0.5f;
        k_matmul_q8_ld(add, &in0, &a, ld, half, d, gs, K_ADD);
        for (int i = 0; i < d; i++) {
          checks++;
          if (fabsf(add[i] - 0.5f - p0[i]) > 1e-4f * (fabsf(p0[i]) + 1) && failures++ < 20)
            fprintf(stderr, "%s [%s]: ADD differs from PARTIAL plus the base at %d\n", name, current, i);
        }
        free(add), free(p0), free(p1), free(both), free(ref), free(mag);
      }
      free(x), free(ws), free(wq), free(w0q), free(w1q), free(w0s), free(w1s), free(xq), free(xs), free(xq0), free(xq1),
          free(xs0), free(xs1);
    }
  }
}

static void count_rows(void *ctx, int r0, int r1) {
  int *seen = ctx;
  for (int i = r0; i < r1; i++)
    seen[i]++;
}

/* LLAMA_GEMV_STAGGER starts team member w at block w of its share and wraps: every output is bitwise the unstaggered
 * one, for every epilogue, fused form and activation form, whatever the number of rows, and each worker reports its whole
 * share to the rows-done callback exactly once. */
static void test_gemv_stagger(void) {
  const int n = 256, rows[] = {1, 3, 4, 5, 13, 100, 257}, previous = k_gemv_stagger_active();
  for (int gs = 32; gs <= 64; gs *= 2)
    for (size_t r = 0; r < sizeof(rows) / sizeof(*rows); r++)
      for (int mode = 0; mode < 2; mode++) {
        const int d = rows[r], groups = n / gs;
        float *x = zeroed(n, sizeof(float)), *ws = zeroed((size_t)3 * d * groups, sizeof(float));
        int8_t *wq = zeroed((size_t)3 * d * n, 1), *xq = zeroed(n, 1);
        float *xs = zeroed(groups, sizeof(float));
        fill(x, n, 3);
        for (size_t i = 0; i < (size_t)3 * d * n; i++)
          wq[i] = (int8_t)(uniform() * 127.0f);
        for (int i = 0; i < 3 * d * groups; i++)
          ws[i] = 0.001f + 0.01f * fabsf(uniform());
        k_quantize(xq, xs, x, n, gs);
        K8Input in = mode ? (K8Input){NULL, NULL, x} : (K8Input){xq, xs, x};
        K8Weight w[3] = {{wq, ws}, {wq + (size_t)d * n, ws + (size_t)d * groups},
                         {wq + (size_t)2 * d * n, ws + (size_t)2 * d * groups}};
        float *out[2][5];
        for (int on = 0; on < 2; on++) {
          k_gemv_stagger(on);
          for (int k = 0; k < 5; k++) {
            out[on][k] = zeroed(3 * (size_t)d, sizeof(float));
            for (int i = 0; i < 3 * d; i++)
              out[on][k][i] = 0.25f * (float)(i % 7);
          }
          k_matmul_q8(out[on][0], &in, w[0], n, d, gs);
          k_matmul_q8_add(out[on][1], &in, w[1], n, d, gs);
          k_matmul_q8_qkv(out[on][2], out[on][2] + d, out[on][2] + 2 * d, &in, w[0], w[1], w[2], n, d, d, gs);
          k_matmul_q8_swiglu(out[on][3], &in, w[0], w[1], n, d, gs);
          k_matmul_q8_ld(out[on][4], &in, &w[2], (size_t)n, n, d, gs, K_PARTIAL);
        }
        k_gemv_stagger(previous);
        for (int k = 0; k < 5; k++) {
          checks++;
          if (memcmp(out[0][k], out[1][k], 3 * (size_t)d * sizeof(float)) && failures++ < 20)
            fprintf(stderr, "gemv stagger [%s]: form %d d=%d gs=%d differs from the unstaggered result\n", current, k, d,
                    gs);
        }
        for (int workers = 1; workers <= 9; workers += 4) {
          float *serial = zeroed(d, sizeof(float));
          int *seen = zeroed(d, sizeof(int));
          k_gemv_stagger(1);
          for (int worker = 0; worker < workers; worker++)
            k_matmul_q8_ld_worker(serial, &in, &w[2], (size_t)n, n, d, gs, K_PARTIAL, worker, workers, NULL, count_rows,
                                  seen);
          k_gemv_stagger(previous);
          checks++;
          if (memcmp(serial, out[0][4], d * sizeof(float)) && failures++ < 20)
            fprintf(stderr, "gemv stagger [%s]: worker entry d=%d workers=%d differs\n", current, d, workers);
          for (int i = 0; i < d; i++) {
            checks++;
            if (seen[i] != 1 && failures++ < 20)
              fprintf(stderr, "gemv stagger [%s]: row %d reported %d times (d=%d workers=%d)\n", current, i, seen[i], d,
                      workers);
          }
          free(serial), free(seen);
        }
        for (int on = 0; on < 2; on++)
          for (int k = 0; k < 5; k++)
            free(out[on][k]);
        free(x), free(ws), free(wq), free(xq), free(xs);
      }
}

/* LLAMA_HYBRID_SPLIT=rows: a weight whose last rows live in a second allocation (the DDR arena) gives bitwise the output
 * of the same weight in one piece, in every fused form and epilogue, at every split (also the ends), with and without
 * the stagger, through the team-less path and through the per-member entry point for every team size and number of
 * members serving the second place; every row is reported done exactly once. */
static void test_gemv_rows(void) {
  const int n = 256, rows[] = {32, 64, 100, 257};
  for (int gs = 32; gs <= 64; gs *= 2)
    for (size_t r = 0; r < sizeof(rows) / sizeof(*rows); r++)
      for (int mode = 0; mode < 2; mode++) {
        const int d = rows[r], groups = n / gs;
        float *x = zeroed(n, sizeof(float)), *ws = zeroed((size_t)2 * d * groups, sizeof(float));
        int8_t *wq = zeroed((size_t)2 * d * n, 1), *xq = zeroed(n, 1);
        float *xs = zeroed(groups, sizeof(float));
        fill(x, n, 3);
        for (size_t i = 0; i < (size_t)2 * d * n; i++)
          wq[i] = (int8_t)(uniform() * 127.0f);
        for (int i = 0; i < 2 * d * groups; i++)
          ws[i] = 0.001f + 0.01f * fabsf(uniform());
        k_quantize(xq, xs, x, n, gs);
        K8Input in = mode ? (K8Input){NULL, NULL, x} : (K8Input){xq, xs, x};
        K8Weight whole[2] = {{wq, ws}, {wq + (size_t)d * n, ws + (size_t)d * groups}};
        float *want[4];
        for (int k = 0; k < 4; k++) {
          want[k] = zeroed(d, sizeof(float));
          for (int i = 0; i < d; i++)
            want[k][i] = 0.25f * (float)(i % 7);
        }
        k_matmul_q8(want[0], &in, whole[0], n, d, gs);
        k_matmul_q8_add(want[1], &in, whole[0], n, d, gs);
        k_matmul_q8_swiglu(want[2], &in, whole[0], whole[1], n, d, gs);
        k_matmul_q8_ld(want[3], &in, &whole[1], (size_t)n, n, d, gs, K_PARTIAL);
        int splits[] = {0, 32, d / 2 / 32 * 32, d - 32, d};
        for (size_t si = 0; si < sizeof(splits) / sizeof(*splits); si++) {
          const int h = splits[si];
          /* the rows from h on, copied to their own allocations, so a wrong base or row index reads other bytes */
          K8Rows rw[2];
          int8_t *hq[2];
          float *hs[2];
          for (int m = 0; m < 2; m++) {
            hq[m] = zeroed((size_t)(d - h) * n + 64, 1);
            hs[m] = zeroed((size_t)(d - h) * groups + 16, sizeof(float));
            memcpy(hq[m], whole[m].q + (size_t)h * n, (size_t)(d - h) * n);
            memcpy(hs[m], whole[m].s + (size_t)h * groups, (size_t)(d - h) * groups * sizeof(float));
            rw[m] = (K8Rows){.lo = whole[m], .hi = {hq[m], hs[m]}, .split = h};
          }
          for (int on = 0; on < 2; on++) {
            k_gemv_stagger(on);
            float *got[3];
            for (int k = 0; k < 3; k++) {
              got[k] = zeroed(d, sizeof(float));
              for (int i = 0; i < d; i++)
                got[k][i] = 0.25f * (float)(i % 7);
            }
            k_matmul_q8_rows(got[0], &in, rw[0], n, d, gs);
            k_matmul_q8_add_rows(got[1], &in, rw[0], n, d, gs);
            k_matmul_q8_swiglu_rows(got[2], &in, rw[0], rw[1], n, d, gs);
            for (int k = 0; k < 3; k++) {
              checks++;
              if (memcmp(got[k], want[k], (size_t)d * sizeof(float)) && failures++ < 20)
                fprintf(stderr, "gemv rows [%s]: form %d d=%d split=%d gs=%d stagger=%d differs from the unsplit result\n",
                        current, k, d, h, gs, on);
            }
            /* qkv: three projections of one input, each split at its own place */
            float *qkv = zeroed(3 * (size_t)d, sizeof(float)), *qkv_want = zeroed(3 * (size_t)d, sizeof(float));
            k_matmul_q8_qkv_rows(qkv, qkv + d, qkv + 2 * d, &in, rw[0], rw[1], rw[0], n, d, d, gs);
            k_matmul_q8_qkv(qkv_want, qkv_want + d, qkv_want + 2 * d, &in, whole[0], whole[1], whole[0], n, d, d, gs);
            checks++;
            if (memcmp(qkv, qkv_want, 3 * (size_t)d * sizeof(float)) && failures++ < 20)
              fprintf(stderr, "gemv rows [%s]: qkv d=%d split=%d gs=%d differs\n", current, d, h, gs);
            free(qkv), free(qkv_want);
            for (int workers = 1; workers <= 9; workers += 2)
              for (int cores = 0; cores <= 4; cores += 2) {
                k_gemv_ddr_cores(cores);
                float *serial = zeroed(d, sizeof(float));
                int *seen = zeroed(d, sizeof(int));
                for (int worker = 0; worker < workers; worker++)
                  k_matmul_q8_ld_worker_rows(serial, &in, &rw[1], (size_t)n, n, d, gs, K_PARTIAL, worker, workers, NULL,
                                             count_rows, seen);
                checks++;
                if (memcmp(serial, want[3], (size_t)d * sizeof(float)) && failures++ < 20)
                  fprintf(stderr, "gemv rows [%s]: worker entry d=%d split=%d workers=%d cores=%d differs\n", current, d,
                          h, workers, cores);
                for (int i = 0; i < d; i++) {
                  checks++;
                  if (seen[i] != 1 && failures++ < 20)
                    fprintf(stderr, "gemv rows [%s]: row %d reported %d times (d=%d split=%d workers=%d cores=%d)\n",
                            current, i, seen[i], d, h, workers, cores);
                }
                free(serial), free(seen);
              }
            k_gemv_ddr_cores(0);
            k_gemv_stagger(0);
            for (int k = 0; k < 3; k++)
              free(got[k]);
          }
          for (int m = 0; m < 2; m++)
            free(hq[m]), free(hs[m]);
        }
        for (int k = 0; k < 4; k++)
          free(want[k]);
        free(x), free(ws), free(wq), free(xq), free(xs);
      }
  /* The members that serve the DDR rows: the default scales 14 of 56 to the team, an explicit count is clamped to leave
   * at least one member for the HBM rows, and a team of one serves both places itself. */
  checks += 5;
  if ((k_gemv_ddr_workers(56) != 14 || k_gemv_ddr_workers(1) != 0 || k_gemv_ddr_workers(2) != 1 ||
       k_gemv_ddr_workers(8) != 2 || k_gemv_ddr_workers(112) != 28) &&
      failures++ < 20)
    fprintf(stderr, "gemv rows [%s]: default DDR members %d %d %d %d %d\n", current, k_gemv_ddr_workers(56),
            k_gemv_ddr_workers(1), k_gemv_ddr_workers(2), k_gemv_ddr_workers(8), k_gemv_ddr_workers(112));
  k_gemv_ddr_cores(100);
  checks++;
  if (k_gemv_ddr_workers(56) != 55 && failures++ < 20)
    fprintf(stderr, "gemv rows [%s]: explicit count not clamped (%d)\n", current, k_gemv_ddr_workers(56));
  k_gemv_ddr_cores(20);
  checks++;
  if (k_gemv_ddr_workers(56) != 20 && failures++ < 20)
    fprintf(stderr, "gemv rows [%s]: explicit count 20 gave %d\n", current, k_gemv_ddr_workers(56));
  k_gemv_ddr_cores(0);
}

static void test_partial(void) {
  test_partial_f32();
  test_partial_q8();
}

/* Two coordinator threads, each bound to its own team, run kernels at the same time. */
typedef struct {
  int socket, failed;
} TwoTeams;

/* The row-parallel forms of batched prefill are the serial row loops, bit for bit, on every instruction set, with and
 * without a worker team: one row, many rows, and an in-place RMSNorm. */
static void test_row_parallel(void) {
  const int sizes[] = {16, 64, 100, 4096};
  const int row_counts[] = {1, 2, 3, 7, 16, 33};
  for (size_t a = 0; a < sizeof(sizes) / sizeof(*sizes); a++)
    for (size_t b = 0; b < sizeof(row_counts) / sizeof(*row_counts); b++) {
      const int size = sizes[a], rows = row_counts[b];
      float *x = zeroed((size_t)rows * size, sizeof(float)), *weight = zeroed(size, sizeof(float));
      float *serial = zeroed((size_t)rows * size, sizeof(float)), *parallel = zeroed((size_t)rows * size, sizeof(float));
      fill(x, (size_t)rows * size, 2.0f), fill(weight, size, 1.0f);
      for (int i = 0; i < rows; i++)
        k_rmsnorm(serial + (size_t)i * size, x + (size_t)i * size, weight, size, 1e-5f);
      k_rmsnorm_rows(parallel, x, weight, rows, size, 1e-5f);
      checks++;
      if (memcmp(serial, parallel, (size_t)rows * size * sizeof(float)) && failures++ < 20)
        fprintf(stderr, "k_rmsnorm_rows [%s] rows=%d size=%d: differs from the serial loop\n", current, rows, size);
      memcpy(parallel, x, (size_t)rows * size * sizeof(float)); /* in place */
      k_rmsnorm_rows(parallel, parallel, weight, rows, size, 1e-5f);
      checks++;
      if (memcmp(serial, parallel, (size_t)rows * size * sizeof(float)) && failures++ < 20)
        fprintf(stderr, "k_rmsnorm_rows in place [%s] rows=%d size=%d: differs from the serial loop\n", current, rows, size);
      free(x), free(weight), free(serial), free(parallel);
    }
  /* RoPE: heads of head_dim 8 and 16 (even), GQA, row strides larger than the data they cover. */
  const int shapes[][3] = {{4, 2, 8}, {8, 2, 16}, {6, 3, 8}};
  for (size_t sh = 0; sh < sizeof(shapes) / sizeof(*shapes); sh++)
    for (size_t b = 0; b < sizeof(row_counts) / sizeof(*row_counts); b++) {
      const int heads = shapes[sh][0], kv_heads = shapes[sh][1], hd = shapes[sh][2], rows = row_counts[b];
      const size_t qs = (size_t)heads * hd + 3, ks = (size_t)kv_heads * hd + 5, ts = (size_t)2 * hd;
      float *q = zeroed(rows * qs, sizeof(float)), *k = zeroed(rows * ks, sizeof(float)), *table = zeroed(rows * ts, sizeof(float));
      float *q2 = zeroed(rows * qs, sizeof(float)), *k2 = zeroed(rows * ks, sizeof(float));
      fill(q, rows * qs, 1.0f), fill(k, rows * ks, 1.0f), fill(table, rows * ts, 1.0f);
      memcpy(q2, q, rows * qs * sizeof(float)), memcpy(k2, k, rows * ks * sizeof(float));
      for (int i = 0; i < rows; i++)
        k_rope_apply(q + i * qs, k + i * ks, table + i * ts, heads, kv_heads, hd);
      k_rope_apply_rows(q2, qs, k2, ks, table, ts, rows, heads, kv_heads, hd);
      checks++;
      if ((memcmp(q, q2, rows * qs * sizeof(float)) || memcmp(k, k2, rows * ks * sizeof(float))) && failures++ < 20)
        fprintf(stderr, "k_rope_apply_rows [%s] heads=%d/%d hd=%d rows=%d: differs from the serial loop\n", current, heads,
                kv_heads, hd, rows);
      free(q), free(k), free(table), free(q2), free(k2);
    }
}

static void *two_team_main(void *p) {
  TwoTeams *job = p;
  kernels_set_thread_team(pool_team(job->socket));
  unsigned long long local = 0x9E3779B97F4A7C15ull * (unsigned)(job->socket + 1);
  const int n = 200, d = 37;
  float *x = zeroed(n, sizeof(float)), *w = zeroed((size_t)n * d, sizeof(float)), *out = zeroed(d, sizeof(float));
  for (int i = 0; i < n; i++)
    x[i] = (float)((local = local * 6364136223846793005ull + 1) >> 40) / (float)(1 << 24) - 0.5f;
  for (int i = 0; i < n * d; i++)
    w[i] = (float)((local = local * 6364136223846793005ull + 1) >> 40) / (float)(1 << 24) - 0.5f;
  for (int rep = 0; rep < 300; rep++) {
    k_matmul(out, x, w, n, d);
    for (int i = 0; i < d; i++) {
      double ref = 0, mag = 0;
      for (int j = 0; j < n; j++) {
        ref += (double)w[(size_t)i * n + j] * x[j];
        mag += fabs((double)w[(size_t)i * n + j] * x[j]);
      }
      if (fabs(out[i] - ref) > 2e-5 * mag + 1e-6)
        job->failed++;
    }
  }
  kernels_set_thread_team(NULL);
  free(x), free(w), free(out);
  return NULL;
}

static void test_two_teams(void) {
  PoolConfig config = {.teams = 2, .threads = {3, 2}, .cpus = {NULL, NULL}};
  pool_start(&config);
  TwoTeams job[2] = {{0, 0}, {1, 0}};
  pthread_t thread;
  assert(!pthread_create(&thread, NULL, two_team_main, &job[1]));
  two_team_main(&job[0]);
  pthread_join(thread, NULL);
  checks += 2;
  if ((job[0].failed || job[1].failed) && failures++ < 20)
    fprintf(stderr, "two teams [%s]: %d + %d results out of budget\n", current, job[0].failed, job[1].failed);
  pool_stop();
}

/* ---- Socket-local attention (B3) ----------------------------------------------------------------------------------
 * Each socket holds K/V for its own KV heads only, in a [position][kv_dim / S] cache, and attends for its own query
 * heads with local counts on its own team. The concatenation of the sockets' outputs must equal the full attention. */
enum { GUARD = 64 };

typedef struct {
  int socket, sockets, heads, kv_heads, hd, pos, seq;
  const float *q, *kc, *vc;
  float *out;    /* the full output; this socket writes its own heads */
  int bad_guard; /* a guard float next to the local cache changed */
  int chunks;    /* chunk count the local work reported */
  int workers;
} SplitJob;

static void *split_main(void *p) {
  SplitJob *job = p;
  kernels_set_thread_team(job->sockets > 1 ? pool_team(job->socket) : NULL);
  int hl = job->heads / job->sockets, kl = job->kv_heads / job->sockets, hd = job->hd, kv_local = kl * hd,
      kv_full = job->kv_heads * hd;
  size_t cells = (size_t)job->seq * kv_local;
  float *k_all = zeroed(cells + 2 * GUARD, sizeof(float)), *v_all = zeroed(cells + 2 * GUARD, sizeof(float));
  for (int i = 0; i < GUARD; i++)
    k_all[i] = v_all[i] = k_all[GUARD + cells + i] = v_all[GUARD + cells + i] = NAN;
  float *kl_cache = k_all + GUARD, *vl_cache = v_all + GUARD;
  for (int t = 0; t < job->seq; t++) {
    memcpy(kl_cache + (size_t)t * kv_local, job->kc + (size_t)t * kv_full + (size_t)job->socket * kv_local,
           kv_local * sizeof(float));
    memcpy(vl_cache + (size_t)t * kv_local, job->vc + (size_t)t * kv_full + (size_t)job->socket * kv_local,
           kv_local * sizeof(float));
  }
  float *att = zeroed((size_t)hl * job->seq, sizeof(float));
  AttentionWork work; /* caller memory: memory_alloc's accounting belongs to one thread */
  float *work_memory = zeroed(k_attention_bytes(hl, kl, hd, job->seq) / sizeof(float) + 1, sizeof(float));
  k_attention_init_in(&work, hl, kl, hd, job->seq, work_memory); /* sized from this thread's team */
  k_attention(job->out + (size_t)job->socket * hl * hd, job->q + (size_t)job->socket * hl * hd, kl_cache, vl_cache, att,
              &work, hl, kl, hd, job->seq, job->pos);
  job->chunks = k_attention_chunk_count(&work, kl, job->pos);
  job->workers = work.workers;
  for (int i = 0; i < GUARD; i++)
    if (!isnan(k_all[i]) || !isnan(k_all[GUARD + cells + i]) || !isnan(v_all[i]) || !isnan(v_all[GUARD + cells + i]))
      job->bad_guard++;
  k_attention_free(&work);
  kernels_set_thread_team(NULL);
  free(k_all), free(v_all), free(att), free(work_memory);
  return NULL;
}

static void test_attention_split(void) {
  const int positions[] = {0, 31, 32, 255, 256, 1000, 4095};
  const int heads = 8, kv_heads = 4, hd = 16, kv_full = kv_heads * hd;
  int team_sizes[8] = {1, 3, 7};
  size_t team_size_count = 3;
  const char *override = getenv("LLAMA_TEST_TEAMS"); /* e.g. "1,56": two teams of that size each */
  if (override && *override) {
    team_size_count = 0;
    for (char *copy = strdup(override), *token = strtok(copy, ","); token && team_size_count < 8;
         token = strtok(NULL, ","))
      team_sizes[team_size_count++] = atoi(token);
  }
  for (size_t ti = 0; ti < team_size_count; ti++) {
    PoolConfig config = {.teams = 2, .threads = {team_sizes[ti], team_sizes[ti]}, .cpus = {NULL, NULL}};
    pool_start(&config);
    for (size_t pi = 0; pi < sizeof(positions) / sizeof(*positions); pi++) {
      int pos = positions[pi], seq = pos + 4;
      float *q = zeroed((size_t)heads * hd, sizeof(float)), *full = zeroed((size_t)heads * hd, sizeof(float));
      float *split = zeroed((size_t)heads * hd, sizeof(float)), *att = zeroed((size_t)heads * seq, sizeof(float));
      float *kc = zeroed((size_t)seq * kv_full, sizeof(float)), *vc = zeroed((size_t)seq * kv_full, sizeof(float));
      fill(q, (size_t)heads * hd, 2);
      fill(kc, (size_t)seq * kv_full, 2);
      fill(vc, (size_t)seq * kv_full, 1);
      AttentionWork work;
      k_attention_init(&work, heads, kv_heads, hd, seq); /* main thread: no team bound, default backend */
      k_attention(full, q, kc, vc, att, &work, heads, kv_heads, hd, seq, pos);
      k_attention_free(&work);
      SplitJob job[2];
      for (int s = 0; s < 2; s++)
        job[s] = (SplitJob){.socket = s,
                            .sockets = 2,
                            .heads = heads,
                            .kv_heads = kv_heads,
                            .hd = hd,
                            .pos = pos,
                            .seq = seq,
                            .q = q,
                            .kc = kc,
                            .vc = vc,
                            .out = split};
      pthread_t thread;
      assert(!pthread_create(&thread, NULL, split_main, &job[1]));
      split_main(&job[0]);
      pthread_join(thread, NULL);
      for (int i = 0; i < heads * hd; i++)
        check("attention.split", split[i], full[i], 2e-5);
      checks += 2;
      if ((job[0].bad_guard || job[1].bad_guard) && failures++ < 20)
        fprintf(stderr, "attention split [%s]: local cache guard overwritten (position %d)\n", current, pos);
      if ((job[0].workers != team_sizes[ti] || job[1].workers != team_sizes[ti] || job[0].chunks < 1 ||
           job[0].chunks != job[1].chunks) &&
          failures++ < 20)
        fprintf(stderr, "attention split [%s]: per-team work sized %d/%d for teams of %d\n", current, job[0].workers,
                job[1].workers, team_sizes[ti]);
      free(q), free(full), free(split), free(att), free(kc), free(vc);
    }
    pool_stop();
  }
}

/* Attention time per socket on socket-local caches at the 8B shape (16 query heads and 4 KV heads per socket), for the
 * B3 cluster check: both sockets attend at the same time on their own teams, at several contexts. */
typedef struct {
  int socket, context, reps;
  atomic_int *go;
  double median;
  int chunks, workers;
} BenchJob;

static int compare_seconds(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static void *bench_attention_main(void *p) {
  BenchJob *job = p;
  kernels_set_thread_team(pool_team(job->socket));
  const int hl = 16, kl = 4, hd = 128, kv_local = kl * hd, seq = job->context + 1;
  float *q = zeroed((size_t)hl * hd, sizeof(float)), *out = zeroed((size_t)hl * hd, sizeof(float));
  float *kc = zeroed((size_t)seq * kv_local, sizeof(float)), *vc = zeroed((size_t)seq * kv_local, sizeof(float));
  float *att = zeroed((size_t)hl * seq, sizeof(float));
  double *samples = zeroed(job->reps, sizeof(double));
  fill(q, (size_t)hl * hd, 2);
  fill(kc, (size_t)seq * kv_local, 2);
  fill(vc, (size_t)seq * kv_local, 1);
  AttentionWork work;
  float *work_memory = zeroed(k_attention_bytes(hl, kl, hd, seq) / sizeof(float) + 1, sizeof(float));
  k_attention_init_in(&work, hl, kl, hd, seq, work_memory);
  atomic_fetch_add(job->go, 1);
  while (atomic_load(job->go) < 2)
    ;
  for (int r = 0; r < job->reps; r++) {
    double start = monotonic_seconds();
    k_attention(out, q, kc, vc, att, &work, hl, kl, hd, seq, job->context);
    samples[r] = monotonic_seconds() - start;
  }
  qsort(samples, (size_t)job->reps, sizeof(double), compare_seconds);
  job->median = samples[job->reps / 2];
  job->chunks = k_attention_chunk_count(&work, kl, job->context);
  job->workers = work.workers;
  k_attention_free(&work);
  kernels_set_thread_team(NULL);
  free(q), free(out), free(kc), free(vc), free(att), free(samples), free(work_memory);
  return NULL;
}

static void bench_attention(int threads) {
  const Topology *topology = numa_topology();
  const int *cpus[2] = {NULL, NULL};
  if (topology->nsockets == 2 && topology->socket[0].cpus && topology->socket[0].ncpus >= threads &&
      topology->socket[1].ncpus >= threads) {
    cpus[0] = topology->socket[0].cpus;
    cpus[1] = topology->socket[1].cpus;
  }
  PoolConfig config = {.teams = 2, .threads = {threads, threads}, .cpus = {cpus[0], cpus[1]}};
  pool_start(&config);
  const int contexts[] = {512, 8192, 32768};
  for (size_t c = 0; c < sizeof(contexts) / sizeof(*contexts); c++) {
    atomic_int go = 0;
    BenchJob job[2];
    for (int s = 0; s < 2; s++)
      job[s] = (BenchJob){.socket = s, .context = contexts[c], .reps = 200, .go = &go};
    pthread_t thread;
    assert(!pthread_create(&thread, NULL, bench_attention_main, &job[1]));
    bench_attention_main(&job[0]);
    pthread_join(thread, NULL);
    printf("{\"kernel\": \"%s\", \"threads\": %d, \"pinned\": %d, \"context\": %d, \"chunks\": %d, "
           "\"median_us\": [%.1f, %.1f]}\n",
           kernels_name(), threads, cpus[0] != NULL, contexts[c], job[0].chunks, job[0].median * 1e6,
           job[1].median * 1e6);
  }
  pool_stop();
}

/* ---- E1: attention test matrix shared by every variant ----------------------------------------------------------------
 * Every attention kernel (today the split-K one, on its public entry points and on the portable driver hook; E2 adds
 * tiled) is one entry of attn_variants and goes through the same cases against one double-precision reference of causal
 * GQA attention that never touches the chunk plan. Per case and variant:
 *   - batched output against the reference (2e-5 absolute, as test_attention), with the output, the scratch and the
 *     split-K workspace poisoned with NaN, and K/V rows beyond the last position poisoned too, so an unwritten row, a
 *     stale partial slot or a read past the context turns into a NaN and fails;
 *   - the same call with the workspace cut to a third of the rows and to one row, so the batch runs in several waves;
 *     the result must equal the single-wave one bit for bit (waves only partition rows);
 *   - causality by perturbation: K/V rows after the position a row may see are replaced with large values, and the
 *     row must not move, bit for bit.
 * Decode (k_attention) is checked at positions on both sides of every chunk boundary. */
typedef void (*AttnBatchFn)(float *out, const float *q, const float *kc, const float *vc, float *att,
                            AttentionWork *work, int heads, int kv_heads, int hd, int seq, int pos0, int m);
typedef void (*AttnDecodeFn)(float *out, const float *q, const float *kc, const float *vc, float *att,
                             AttentionWork *work, int heads, int kv_heads, int hd, int seq, int position);
typedef struct {
  const char *name;
  AttnBatchFn batch;
  AttnDecodeFn decode;
  int vs_split; /* also compared with the split kernel (variant 0) at 1e-5, the E2 gate */
  int decode_only; /* a single-row variant (E5): the batched cases run it for m = 1 only */
} AttnVariant;

/* The driver with the portable chunk kernel runs the batched code (waves, row blocks, merge) on any host; it needs
 * m > 1, and a single row is the decode path anyway. */
static void attn_portable_batch(float *out, const float *q, const float *kc, const float *vc, float *att,
                                AttentionWork *work, int heads, int kv_heads, int hd, int seq, int pos0, int m) {
  if (m > 1)
    k_attention_batch_portable(out, q, kc, vc, work, heads, kv_heads, hd, seq, pos0, m);
  else
    k_attention(out, q, kc, vc, att, work, heads, kv_heads, hd, seq, pos0);
}

/* E2: LLAMA_ATTN=tiled for one call. On a build without AVX-512 (or the scalar level) it hands the call to split with a
 * one-time note, so the same expectations hold there. */
static void attn_tiled_batch(float *out, const float *q, const float *kc, const float *vc, float *att,
                             AttentionWork *work, int heads, int kv_heads, int hd, int seq, int pos0, int m) {
  assert(kernels_attention_select("tiled", NULL));
  k_attention_batch(out, q, kc, vc, att, work, heads, kv_heads, hd, seq, pos0, m);
  assert(kernels_attention_select("split", NULL));
}

/* E5: LLAMA_ATTN_DECODE=group for one call; like tiled it hands the call to split with a one-time note where AVX-512
 * is not active. A single row goes through the decode path, which is where the knob acts. */
static void attn_group_decode(float *out, const float *q, const float *kc, const float *vc, float *att,
                              AttentionWork *work, int heads, int kv_heads, int hd, int seq, int position) {
  assert(kernels_attention_select(NULL, "group"));
  k_attention(out, q, kc, vc, att, work, heads, kv_heads, hd, seq, position);
  assert(kernels_attention_select(NULL, "split"));
}

static void attn_group_batch(float *out, const float *q, const float *kc, const float *vc, float *att,
                             AttentionWork *work, int heads, int kv_heads, int hd, int seq, int pos0, int m) {
  assert(m == 1);
  attn_group_decode(out, q, kc, vc, att, work, heads, kv_heads, hd, seq, pos0);
}

static const AttnVariant attn_variants[] = {
    {"split", k_attention_batch, k_attention, 0, 0},
    {"split-portable", attn_portable_batch, k_attention, 0, 0},
    {"tiled", attn_tiled_batch, k_attention, 1, 0},
    {"group", attn_group_batch, attn_group_decode, 1, 1},
};
enum { ATTN_VARIANTS = sizeof(attn_variants) / sizeof(*attn_variants), ATTN_GUARD_ROWS = 4 };

static void poison(float *x, size_t n) {
  for (size_t i = 0; i < n; i++)
    x[i] = NAN;
}

static void expect(int ok, const char *format, ...) __attribute__((format(printf, 2, 3)));
static void expect(int ok, const char *format, ...) {
  checks++;
  if (ok || failures++ >= 20)
    return;
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
}

/* Causal GQA attention of one query row at `position` (it sees positions 0..position) in double precision. */
static void attention_reference(const float *q, const float *kc, const float *vc, int heads, int kv_heads, int hd,
                                int position, double *out, double *scores) {
  int kv_mul = heads / kv_heads, kv_dim = kv_heads * hd;
  for (int h = 0; h < heads; h++) {
    const float *kh = kc + (size_t)(h / kv_mul) * hd, *vh = vc + (size_t)(h / kv_mul) * hd;
    double max = -INFINITY, sum = 0;
    for (int t = 0; t <= position; t++) {
      double dot = 0;
      for (int i = 0; i < hd; i++)
        dot += (double)q[h * hd + i] * kh[(size_t)t * kv_dim + i];
      scores[t] = dot / sqrt(hd);
      if (scores[t] > max)
        max = scores[t];
    }
    for (int t = 0; t <= position; t++)
      sum += scores[t] = exp(scores[t] - max);
    for (int i = 0; i < hd; i++) {
      double acc = 0;
      for (int t = 0; t <= position; t++)
        acc += scores[t] / sum * vh[(size_t)t * kv_dim + i];
      out[h * hd + i] = acc;
    }
  }
}

typedef struct {
  int kv_heads, group, hd;
} AttnShape;

/* When set, the workspace is sized for this many workers instead of the team's, so a single-threaded test still runs the
 * multi-chunk plans of a 56-worker socket (chunk edges inside a row block, rows with no position in a chunk). */
static int attn_force_workers;
static double matrix_seconds_long;                /* time of the long decode contexts */
static double worst_vs_reference, worst_vs_split; /* largest error of the variants flagged vs_split, printed at the end */

/* One batched case: m rows at pos0..pos0+m-1 on `shape`, through every variant. budget_mib > 0 sets
 * the workspace budget (k_attention_set_workspace_mib) for the sizing instead of cutting the capacity by hand. */
static void attention_batch_case(AttnShape shape, int m, int pos0, int budget_mib) {
  int heads = shape.kv_heads * shape.group, kv_heads = shape.kv_heads, hd = shape.hd;
  int seq = pos0 + m, rows = seq + ATTN_GUARD_ROWS;
  size_t dim = (size_t)heads * hd, kv_dim = (size_t)kv_heads * hd;
  float *q = zeroed(m * dim, sizeof(float)), *kc = zeroed(rows * kv_dim, sizeof(float));
  float *vc = zeroed(rows * kv_dim, sizeof(float)), *att = zeroed((size_t)heads * seq, sizeof(float));
  float *got = zeroed(m * dim, sizeof(float)), *first = zeroed(m * dim, sizeof(float));
  float *again = zeroed(m * dim, sizeof(float)), *kp = zeroed(rows * kv_dim, sizeof(float));
  float *vp = zeroed(rows * kv_dim, sizeof(float)), *split_first = zeroed(m * dim, sizeof(float));
  double *ref = zeroed(m * dim, sizeof(double)), *scores = zeroed(seq, sizeof(double));
  fill(q, m * dim, 2), fill(kc, (size_t)seq * kv_dim, 2), fill(vc, (size_t)seq * kv_dim, 1);
  poison(kc + (size_t)seq * kv_dim, ATTN_GUARD_ROWS * kv_dim);
  poison(vc + (size_t)seq * kv_dim, ATTN_GUARD_ROWS * kv_dim);
  for (int i = 0; i < m; i++)
    attention_reference(q + i * dim, kc, vc, heads, kv_heads, hd, pos0 + i, ref + i * dim, scores);
  char label[160];
  if (budget_mib > 0)
    k_attention_set_workspace_mib(budget_mib);
  AttentionWork work;
  k_attention_init_rows(&work, heads, kv_heads, hd, seq, m, NULL);
  if (budget_mib > 0)
    k_attention_set_workspace_mib(-1);
  if (attn_force_workers) {
    k_attention_free(&work);
    int by_threads = (attn_force_workers + kv_heads - 1) / kv_heads, by_length = (seq + 255) / 256;
    work.workers = attn_force_workers;
    work.max_chunks = by_threads > by_length ? by_threads : by_length;
    work.capacity = (size_t)m * heads * work.max_chunks * (hd + 2);
    work.partial = zeroed(work.capacity, sizeof(float));
    work.external = 1;
  }
  size_t full_capacity = work.capacity, one_row = (size_t)heads * work.max_chunks * (hd + 2);
  if (budget_mib > 0)
    expect(full_capacity / one_row >= 1 && full_capacity / one_row < (size_t)m,
           "attention matrix [%s] budget %d MiB: %zu rows of partials for m=%d, expected 2+ waves\n", current,
           budget_mib, full_capacity / one_row, m);
  for (int v = 0; v < ATTN_VARIANTS; v++) {
    const AttnVariant *variant = &attn_variants[v];
    if (variant->decode_only && m != 1)
      continue;
    /* pass 0: as sized; pass 1: a third of the rows per wave; pass 2: one row per wave (short batches only) */
    for (int pass = 0; pass < 3; pass++) {
      if (pass && (budget_mib > 0 || m < 3 || (pass == 2 && m > 32)))
        continue;
      size_t rows_per_wave = pass == 1 ? (size_t)(m / 3 > 0 ? m / 3 : 1) : 1;
      work.capacity = pass ? (rows_per_wave * one_row < full_capacity ? rows_per_wave * one_row : full_capacity)
                           : full_capacity;
      snprintf(label, sizeof(label), "attention matrix %s g=%d hd=%d m=%d pos0=%d pass=%d", variant->name, shape.group,
               hd, m, pos0, pass);
      poison(got, m * dim), poison(att, (size_t)heads * seq), poison(work.partial, full_capacity);
      variant->batch(got, q, kc, vc, att, &work, heads, kv_heads, hd, seq, pos0, m);
      for (size_t i = 0; i < m * dim; i++) {
        check(label, got[i], ref[i], 2e-5);
        if (variant->vs_split && strcmp(kernels_name(), "scalar") && fabs(got[i] - ref[i]) > worst_vs_reference)
          worst_vs_reference = fabs(got[i] - ref[i]);
      }
      if (pass == 0) {
        memcpy(first, got, m * dim * sizeof(float));
        if (v == 0)
          memcpy(split_first, got, m * dim * sizeof(float));
        if (variant->vs_split)
          for (size_t i = 0; i < m * dim; i++) {
            check(label, got[i], split_first[i], 1e-5); /* "vs split", the second E2 gate */
            if (strcmp(kernels_name(), "scalar") && fabs(got[i] - split_first[i]) > worst_vs_split)
              worst_vs_split = fabs(got[i] - split_first[i]);
          }
      } else
        expect(!memcmp(got, first, m * dim * sizeof(float)), "%s [%s]: waves changed the result bit for bit\n", label,
               current);
    }
    work.capacity = full_capacity;
    const int probes[] = {0, m / 2, m - 1};
    for (int pr = 0; pr < 3; pr++) {
      int i = probes[pr], later = pos0 + i + 1;
      if (later >= seq || (pr && i == probes[pr - 1]))
        continue;
      memcpy(kp, kc, rows * kv_dim * sizeof(float)), memcpy(vp, vc, rows * kv_dim * sizeof(float));
      for (size_t t = (size_t)later * kv_dim; t < (size_t)seq * kv_dim; t++)
        kp[t] = 40 * uniform(), vp[t] = 40 * uniform();
      poison(again, m * dim), poison(att, (size_t)heads * seq), poison(work.partial, full_capacity);
      variant->batch(again, q, kp, vp, att, &work, heads, kv_heads, hd, seq, pos0, m);
      expect(!memcmp(again + i * dim, first + i * dim, dim * sizeof(float)),
             "attention matrix %s causality [%s] g=%d hd=%d m=%d pos0=%d row=%d: later K/V changed the row\n",
             variant->name, current, shape.group, hd, m, pos0, i);
    }
  }
  k_attention_free(&work);
  if (attn_force_workers)
    free(work.partial);
  free(q), free(kc), free(vc), free(att), free(got), free(first), free(again), free(kp), free(vp), free(ref),
      free(scores), free(split_first);
}

/* Single-row attention at `positions`, on a context whose rows after the position are NaN, through every variant. */
static void attention_decode_case(AttnShape shape, const int *positions, int count) {
  int heads = shape.kv_heads * shape.group, kv_heads = shape.kv_heads, hd = shape.hd, last = positions[count - 1];
  int seq = last + 1 + ATTN_GUARD_ROWS;
  size_t dim = (size_t)heads * hd, kv_dim = (size_t)kv_heads * hd;
  float *q = zeroed(dim, sizeof(float)), *kc = zeroed(seq * kv_dim, sizeof(float));
  float *vc = zeroed(seq * kv_dim, sizeof(float)), *kp = zeroed(seq * kv_dim, sizeof(float));
  float *vp = zeroed(seq * kv_dim, sizeof(float)), *att = zeroed((size_t)heads * seq, sizeof(float));
  float *got = zeroed(dim, sizeof(float)), *second = zeroed(dim, sizeof(float));
  float *split_got = zeroed(dim, sizeof(float));
  double *ref = zeroed(dim, sizeof(double)), *scores = zeroed(seq, sizeof(double));
  fill(q, dim, 2), fill(kc, (size_t)seq * kv_dim, 2), fill(vc, (size_t)seq * kv_dim, 1);
  AttentionWork work;
  k_attention_init(&work, heads, kv_heads, hd, seq);
  char label[160];
  for (int p = 0; p < count; p++) {
    int position = positions[p];
    attention_reference(q, kc, vc, heads, kv_heads, hd, position, ref, scores);
    memcpy(kp, kc, seq * kv_dim * sizeof(float)), memcpy(vp, vc, seq * kv_dim * sizeof(float));
    poison(kp + (size_t)(position + 1) * kv_dim, (size_t)(seq - position - 1) * kv_dim);
    poison(vp + (size_t)(position + 1) * kv_dim, (size_t)(seq - position - 1) * kv_dim);
    for (int v = 0; v < ATTN_VARIANTS; v++) {
      snprintf(label, sizeof(label), "attention matrix %s decode g=%d hd=%d pos=%d", attn_variants[v].name,
               shape.group, hd, position);
      poison(got, dim), poison(att, (size_t)heads * seq), poison(work.partial, work.capacity);
      attn_variants[v].decode(got, q, kp, vp, att, &work, heads, kv_heads, hd, seq, position);
      for (size_t i = 0; i < dim; i++)
        check(label, got[i], ref[i], 2e-5);
      if (v == 0)
        memcpy(split_got, got, dim * sizeof(float));
      else if (attn_variants[v].vs_split)
        for (size_t i = 0; i < dim; i++) {
          check(label, got[i], split_got[i], 1e-5); /* "vs split" */
          if (strcmp(kernels_name(), "scalar") && fabs(got[i] - split_got[i]) > worst_vs_split)
            worst_vs_split = fabs(got[i] - split_got[i]);
        }
      /* the rows after the position are NaN above; the result must not move when they hold real data */
      poison(second, dim), poison(att, (size_t)heads * seq), poison(work.partial, work.capacity);
      attn_variants[v].decode(second, q, kc, vc, att, &work, heads, kv_heads, hd, seq, position);
      expect(!memcmp(got, second, dim * sizeof(float)), "%s [%s]: later K/V rows changed the result\n", label,
             current);
    }
  }
  k_attention_free(&work);
  free(q), free(kc), free(vc), free(kp), free(vp), free(att), free(got), free(second), free(ref), free(scores),
      free(split_got);
}

/* E2: a partial with no position in it (value sum 0, maximum -FLT_MAX, sum 0) must not move the merge, whatever the
 * value-sum floats hold, at any chunk index. The merge of the real chunks alone is the expectation, bit for bit. */
static void test_attention_empty_partial(void) {
  enum { HD = 24, SLOT = HD + 2, CHUNKS = 5 };
  float real[CHUNKS][SLOT], mixed[CHUNKS * 2 + 1][SLOT], want[HD], got[HD];
  fill(&real[0][0], CHUNKS * SLOT, 3);
  for (int c = 0; c < CHUNKS; c++)
    real[c][HD + 1] = 1.0f + 4.0f * fabsf(real[c][HD + 1]); /* a real chunk has sum >= 1 */
  k_attention_merge_partials(&real[0][0], SLOT, CHUNKS, HD, want);
  for (int style = 0; style < 3; style++) {
    int n = 0;
    for (int c = 0; c <= CHUNKS; c++) { /* an empty slot before every real one and one at the end */
      for (int i = 0; i < HD; i++)
        mixed[n][i] = style == 0 ? 0.0f : style == 1 ? NAN : 1e30f;
      mixed[n][HD] = -FLT_MAX, mixed[n][HD + 1] = 0.0f;
      n++;
      if (c < CHUNKS)
        memcpy(mixed[n++], real[c], sizeof(real[c]));
    }
    memset(got, 0xAB, sizeof(got));
    k_attention_merge_partials(&mixed[0][0], SLOT, n, HD, got);
    checks++;
    if (memcmp(got, want, sizeof(want)) && failures++ < 20)
      fprintf(stderr, "empty partial merge [%s] style=%d: moved the result\n", current, style);
  }
  /* the double reference of the same merge, to make sure the expectation itself is the softmax */
  double max = -1e30, sum = 0;
  for (int c = 0; c < CHUNKS; c++)
    max = fmax(max, real[c][HD]);
  for (int c = 0; c < CHUNKS; c++)
    sum += exp(real[c][HD] - max) * real[c][HD + 1];
  for (int i = 0; i < HD; i++) {
    double acc = 0;
    for (int c = 0; c < CHUNKS; c++)
      acc += exp(real[c][HD] - max) * real[c][i];
    check("empty partial merge reference", want[i], acc / sum, 1e-5);
  }
}

/* E2: the multi-chunk plans of a many-worker socket on any host: every position of a short context is a chunk edge for
 * some row block, so rows with fewer valid chunks than their block (empty partials) occur at every shape. */
static void test_attention_forced_plans(void) {
  const int workers[] = {3, 6, 56};
  for (size_t w = 0; w < sizeof(workers) / sizeof(*workers); w++) {
    attn_force_workers = workers[w];
    for (int pos0 = 0; pos0 <= 70; pos0 += (pos0 < 40 ? 1 : 3)) {
      attention_batch_case((AttnShape){2, 4, 64}, 7, pos0, 0);
      attention_batch_case((AttnShape){2, 1, 40}, 9, pos0, 0);
      attention_batch_case((AttnShape){2, 4, 64}, 1, pos0, 0); /* E5: a single row on a many-worker plan */
      attention_batch_case((AttnShape){2, 2, 40}, 1, pos0, 0);
    }
    attention_batch_case((AttnShape){2, 2, 64}, 33, 100, 0);
    attention_batch_case((AttnShape){2, 8, 64}, 5, 250, 0);
  }
  attn_force_workers = 0;
}

/* Tiled attention on many-worker plans against the reference, and the task hook (the benchmark's path) changes nothing
 * bit for bit, also with several waves. (E2 also swept the block size and the claim order; both fixed since F5.) */
static atomic_long hook_tasks, hook_positions;
static void count_task(void *ctx, const AttentionTaskInfo *info) {
  (void)ctx;
  atomic_fetch_add(&hook_tasks, 1);
  atomic_fetch_add(&hook_positions, info->positions);
}

static void test_attention_tiled_hook(void) {
  attn_force_workers = 6;
  attention_batch_case((AttnShape){2, 4, 64}, 13, 61, 0);
  attention_batch_case((AttnShape){2, 2, 40}, 9, 33, 0);
  attention_batch_case((AttnShape){2, 8, 64}, 5, 250, 0);
  attn_force_workers = 0;
  const AttnShape shapes[] = {{2, 4, 64}, {2, 2, 128}, {2, 1, 40}};
  for (size_t sh = 0; sh < sizeof(shapes) / sizeof(*shapes); sh++) {
    int heads = shapes[sh].kv_heads * shapes[sh].group, kv_heads = shapes[sh].kv_heads, hd = shapes[sh].hd;
    int m = 21, pos0 = 200, seq = pos0 + m, workers = 6;
    size_t dim = (size_t)heads * hd, kv_dim = (size_t)kv_heads * hd;
    float *q = zeroed(m * dim, sizeof(float)), *kc = zeroed(seq * kv_dim, sizeof(float));
    float *vc = zeroed(seq * kv_dim, sizeof(float)), *att = zeroed((size_t)heads * seq, sizeof(float));
    float *want = zeroed(m * dim, sizeof(float)), *got = zeroed(m * dim, sizeof(float));
    fill(q, m * dim, 2), fill(kc, seq * kv_dim, 2), fill(vc, seq * kv_dim, 1);
    AttentionWork work;
    k_attention_init_rows(&work, heads, kv_heads, hd, seq, m, NULL);
    k_attention_free(&work);
    work.workers = workers, work.max_chunks = (workers + kv_heads - 1) / kv_heads > (seq + 255) / 256
                                                  ? (workers + kv_heads - 1) / kv_heads
                                                  : (seq + 255) / 256;
    size_t one_row = (size_t)heads * work.max_chunks * (hd + 2);
    work.partial = zeroed(m * one_row, sizeof(float)), work.external = 1;
    for (int pass = 0; pass < 2; pass++) {
      work.capacity = pass ? 3 * one_row : m * one_row; /* pass 1: waves of three rows */
      assert(kernels_attention_select("tiled", NULL));
      k_attention_batch(want, q, kc, vc, att, &work, heads, kv_heads, hd, seq, pos0, m);
      memset(got, 0xAB, m * dim * sizeof(float));
      atomic_store(&hook_tasks, 0), atomic_store(&hook_positions, 0);
      k_attention_set_task_hook(count_task, NULL); /* the benchmark's path through the tasks */
      k_attention_batch(got, q, kc, vc, att, &work, heads, kv_heads, hd, seq, pos0, m);
      k_attention_set_task_hook(NULL, NULL);
      /* every row sees pos0 + i + 1 positions and each of them is scored by one task per KV head */
      long visible = 0;
      for (int i = 0; i < m; i++)
        visible += (long)(pos0 + i + 1) * kv_heads;
      int ran_tiled = strcmp(kernels_name(), "scalar") != 0; /* the scalar level hands tiled to split, which has no hook */
      checks += 2;
      if (ran_tiled && atomic_load(&hook_tasks) <= 0 && failures++ < 20)
        fprintf(stderr, "tiled hook [%s]: no task reported\n", current);
      if (ran_tiled && atomic_load(&hook_positions) != visible && failures++ < 20)
        fprintf(stderr, "tiled hook [%s]: %ld positions reported, %ld expected\n", current,
                (long)atomic_load(&hook_positions), visible);
      assert(kernels_attention_select("split", NULL));
      checks++;
      if (memcmp(got, want, m * dim * sizeof(float)) && failures++ < 20)
        fprintf(stderr, "tiled hook [%s] hd=%d g=%d pass=%d: the hook changed the result\n", current, hd,
                shapes[sh].group, pass);
    }
    free(work.partial), free(q), free(kc), free(vc), free(att), free(want), free(got);
  }
}

/* E5: long single-row contexts through every decode variant (the chunk plan is at its 256-position chunks there), up to
 * the 32767 positions of the longest campaign context. */
static void test_attention_decode_long(void) {
  double started = monotonic_seconds();
  const int positions[] = {1023, 4095, 8191, 32767};
  attention_decode_case((AttnShape){2, 4, 128}, positions, sizeof(positions) / sizeof(*positions));
  attention_decode_case((AttnShape){2, 8, 40}, positions, 3);
  attention_decode_case((AttnShape){2, 1, 64}, positions + 1, 2);
  matrix_seconds_long += monotonic_seconds() - started;
}


/* E5.2: the decode kernels on a head-major K/V layout [kv_head][position][head_dim] (k_attention_layout with a position
 * stride of head_dim and a head base wider than the head's rows). The same values as the rows layout, so:
 *   - split and group each give the result of the rows layout BIT FOR BIT (the arithmetic does not depend on where a row
 *     is, only on its values), here also with padding between the heads;
 *   - against the double reference at 2e-5, and group against split at 1e-5 (as the rows-layout cases);
 *   - rows after the position and the padding are NaN, so a read past the context turns into a NaN and fails. */
static void attention_head_layout_case(AttnShape shape, const int *positions, int count) {
  int heads = shape.kv_heads * shape.group, kv_heads = shape.kv_heads, hd = shape.hd, last = positions[count - 1];
  int seq = last + 1 + ATTN_GUARD_ROWS;
  size_t dim = (size_t)heads * hd, kv_dim = (size_t)kv_heads * hd;
  size_t head_base = ((size_t)seq + 3) * hd;
  float *q = zeroed(dim, sizeof(float)), *kc = zeroed(seq * kv_dim, sizeof(float));
  float *vc = zeroed(seq * kv_dim, sizeof(float)), *kp = zeroed(seq * kv_dim, sizeof(float));
  float *vp = zeroed(seq * kv_dim, sizeof(float)), *att = zeroed((size_t)heads * seq, sizeof(float));
  float *kh = zeroed(head_base * kv_heads, sizeof(float)), *vh = zeroed(head_base * kv_heads, sizeof(float));
  float *rows = zeroed(dim, sizeof(float)), *head = zeroed(dim, sizeof(float)), *split_head = zeroed(dim, sizeof(float));
  double *ref = zeroed(dim, sizeof(double)), *scores = zeroed(seq, sizeof(double));
  fill(q, dim, 2), fill(kc, (size_t)seq * kv_dim, 2), fill(vc, (size_t)seq * kv_dim, 1);
  AttentionWork work;
  k_attention_init(&work, heads, kv_heads, hd, seq);
  char label[160];
  for (int p = 0; p < count; p++) {
    int position = positions[p];
    attention_reference(q, kc, vc, heads, kv_heads, hd, position, ref, scores);
    memcpy(kp, kc, seq * kv_dim * sizeof(float)), memcpy(vp, vc, seq * kv_dim * sizeof(float));
    poison(kp + (size_t)(position + 1) * kv_dim, (size_t)(seq - position - 1) * kv_dim);
    poison(vp + (size_t)(position + 1) * kv_dim, (size_t)(seq - position - 1) * kv_dim);
    poison(kh, head_base * kv_heads), poison(vh, head_base * kv_heads);
    for (int t = 0; t < seq; t++)
      for (int g = 0; g < kv_heads; g++) {
        memcpy(kh + g * head_base + (size_t)t * hd, kp + (size_t)t * kv_dim + (size_t)g * hd, hd * sizeof(float));
        memcpy(vh + g * head_base + (size_t)t * hd, vp + (size_t)t * kv_dim + (size_t)g * hd, hd * sizeof(float));
      }
    const char *kernel[2] = {"split", "group"};
    for (int k = 0; k < 2; k++) {
      assert(kernels_attention_select(NULL, kernel[k]));
      snprintf(label, sizeof(label), "attention head layout %s g=%d hd=%d pos=%d", kernel[k], shape.group, hd, position);
      poison(rows, dim), poison(att, (size_t)heads * seq), poison(work.partial, work.capacity);
      k_attention(rows, q, kp, vp, att, &work, heads, kv_heads, hd, seq, position);
      poison(head, dim), poison(att, (size_t)heads * seq), poison(work.partial, work.capacity);
      k_attention_layout(head, q, kh, vh, att, &work, heads, kv_heads, hd, seq, position, (size_t)hd, head_base);
      for (size_t i = 0; i < dim; i++)
        check(label, head[i], ref[i], 2e-5);
      expect(!memcmp(rows, head, dim * sizeof(float)), "%s [%s]: head-major result differs from rows bit for bit\n",
             label, current);
      if (k == 0)
        memcpy(split_head, head, dim * sizeof(float));
      else
        for (size_t i = 0; i < dim; i++)
          check(label, head[i], split_head[i], 1e-5); /* group against split, both head-major */
    }
    assert(kernels_attention_select(NULL, "split"));
  }
  k_attention_free(&work);
  free(q), free(kc), free(vc), free(kp), free(vp), free(att), free(kh), free(vh), free(rows), free(head),
      free(split_head), free(ref), free(scores);
}

static void test_attention_head_layout(int with_long) {
  const int boundary[] = {0, 1, 31, 32, 33, 255, 256, 257, 511, 512, 513};
  const AttnShape shapes[] = {{2, 1, 64}, {2, 4, 64}, {2, 4, 128}, {2, 8, 40}, {4, 4, 128}};
  for (size_t i = 0; i < sizeof(shapes) / sizeof(*shapes); i++)
    attention_head_layout_case(shapes[i], boundary, sizeof(boundary) / sizeof(*boundary));
  const int long_positions[] = {1023, 8191, 32767};
  if (with_long)
    attention_head_layout_case((AttnShape){2, 4, 128}, long_positions, sizeof(long_positions) / sizeof(*long_positions));
}

/* E5.4: batched prefill attention (split, split-portable, tiled) on the head-major K/V layout, through both the
 * explicit strides (k_attention_batch_layout with padding between the heads) and the engine helpers
 * (LLAMA_KV_LAYOUT=head: k_kv_store, k_attention_batch_kv). The result must equal the rows layout BIT FOR BIT on the
 * same values, in one wave and in several, on the chunk plan of `workers` many workers when forced. K/V positions at
 * and after the context, and the padding between the heads, are NaN, so a read of anything the rows layout would not
 * read fails. */
static void attention_batch_head_case(AttnShape shape, int m, int pos0) {
  int heads = shape.kv_heads * shape.group, kv_heads = shape.kv_heads, hd = shape.hd;
  int seq = pos0 + m, cap = seq + ATTN_GUARD_ROWS;
  size_t dim = (size_t)heads * hd, kv_dim = (size_t)kv_heads * hd, pad_base = ((size_t)cap + 3) * hd;
  float *q = zeroed(m * dim, sizeof(float)), *kc = zeroed((size_t)cap * kv_dim, sizeof(float));
  float *vc = zeroed((size_t)cap * kv_dim, sizeof(float)), *att = zeroed((size_t)heads * cap, sizeof(float));
  float *want = zeroed(m * dim, sizeof(float)), *got = zeroed(m * dim, sizeof(float));
  float *kp = zeroed(pad_base * kv_heads, sizeof(float)), *vp = zeroed(pad_base * kv_heads, sizeof(float));
  float *kh = zeroed((size_t)cap * kv_dim, sizeof(float)), *vh = zeroed((size_t)cap * kv_dim, sizeof(float));
  fill(q, m * dim, 2), fill(kc, (size_t)seq * kv_dim, 2), fill(vc, (size_t)seq * kv_dim, 1);
  poison(kc + (size_t)seq * kv_dim, (size_t)ATTN_GUARD_ROWS * kv_dim);
  poison(vc + (size_t)seq * kv_dim, (size_t)ATTN_GUARD_ROWS * kv_dim);
  /* explicit strides, head bases wider than the heads' rows */
  poison(kp, pad_base * kv_heads), poison(vp, pad_base * kv_heads);
  for (int t = 0; t < seq; t++)
    for (int g = 0; g < kv_heads; g++) {
      memcpy(kp + g * pad_base + (size_t)t * hd, kc + (size_t)t * kv_dim + (size_t)g * hd, hd * sizeof(float));
      memcpy(vp + g * pad_base + (size_t)t * hd, vc + (size_t)t * kv_dim + (size_t)g * hd, hd * sizeof(float));
    }
  /* the engine layout: seq_len positions per head, written by k_kv_store; positions past the context stay NaN */
  poison(kh, (size_t)cap * kv_dim), poison(vh, (size_t)cap * kv_dim);
  assert(kernels_kv_layout_select("head"));
  k_kv_store(kh, kc, kv_dim, 0, seq, kv_heads, hd, cap);
  k_kv_store(vh, vc, kv_dim, 0, seq, kv_heads, hd, cap);
  size_t strides[2];
  k_kv_strides(kv_heads, hd, cap, &strides[0], &strides[1]);
  expect(strides[0] == (size_t)hd && strides[1] == (size_t)cap * hd, "k_kv_strides [%s]: head layout strides wrong\n",
         current);
  assert(kernels_kv_layout_select("rows"));
  k_kv_strides(kv_heads, hd, cap, &strides[0], &strides[1]);
  expect(strides[0] == kv_dim && strides[1] == (size_t)hd, "k_kv_strides [%s]: rows layout strides wrong\n", current);
  AttentionWork work;
  k_attention_init_rows(&work, heads, kv_heads, hd, cap, m, NULL);
  if (attn_force_workers) {
    k_attention_free(&work);
    int by_threads = (attn_force_workers + kv_heads - 1) / kv_heads, by_length = (cap + 255) / 256;
    work.workers = attn_force_workers;
    work.max_chunks = by_threads > by_length ? by_threads : by_length;
    work.capacity = (size_t)m * heads * work.max_chunks * (hd + 2);
    work.partial = zeroed(work.capacity, sizeof(float));
    work.external = 1;
  }
  size_t full = work.capacity, one_row = (size_t)heads * work.max_chunks * (hd + 2);
  const char *names[3] = {"split", "split-portable", "tiled"};
  char label[160];
  for (int v = 0; v < 3; v++) {
    if (v == 1 && m < 2)
      continue;
    assert(kernels_attention_select(v == 2 ? "tiled" : "split", NULL));
    for (int pass = 0; pass < 3; pass++) {
      if (pass && (m < 3 || (pass == 2 && m > 32)))
        continue;
      size_t rows_per_wave = pass == 1 ? (size_t)(m / 3 > 0 ? m / 3 : 1) : 1;
      work.capacity = pass ? (rows_per_wave * one_row < full ? rows_per_wave * one_row : full) : full;
      snprintf(label, sizeof(label), "attention head layout batch %s g=%d hd=%d m=%d pos0=%d pass=%d", names[v],
               shape.group, hd, m, pos0, pass);
      poison(want, m * dim), poison(att, (size_t)heads * cap), poison(work.partial, full);
      if (v == 1)
        k_attention_batch_portable(want, q, kc, vc, &work, heads, kv_heads, hd, cap, pos0, m);
      else
        k_attention_batch(want, q, kc, vc, att, &work, heads, kv_heads, hd, cap, pos0, m);
      poison(got, m * dim), poison(att, (size_t)heads * cap), poison(work.partial, full);
      if (v == 1)
        k_attention_batch_portable_layout(got, q, kp, vp, &work, heads, kv_heads, hd, cap, pos0, m, (size_t)hd,
                                          pad_base);
      else
        k_attention_batch_layout(got, q, kp, vp, att, &work, heads, kv_heads, hd, cap, pos0, m, (size_t)hd, pad_base);
      expect(!memcmp(got, want, m * dim * sizeof(float)), "%s [%s]: head-major result differs from rows bit for bit\n",
             label, current);
      if (v != 1) { /* the engine entry point with the layout selected */
        assert(kernels_kv_layout_select("head"));
        poison(got, m * dim), poison(att, (size_t)heads * cap), poison(work.partial, full);
        k_attention_batch_kv(got, q, kh, vh, att, &work, heads, kv_heads, hd, cap, pos0, m);
        assert(kernels_kv_layout_select("rows"));
        expect(!memcmp(got, want, m * dim * sizeof(float)),
               "%s [%s]: k_attention_batch_kv (head) differs from rows bit for bit\n", label, current);
      }
    }
    assert(kernels_attention_select("split", NULL));
  }
  work.capacity = full;
  k_attention_free(&work);
  if (attn_force_workers)
    free(work.partial);
  free(q), free(kc), free(vc), free(att), free(want), free(got), free(kp), free(vp), free(kh), free(vh);
}

static void test_attention_batch_head_layout(int with_forced) {
  const AttnShape shapes[] = {{2, 1, 64}, {2, 4, 64}, {2, 4, 128}, {2, 8, 40}, {4, 2, 128}};
  const int ms[] = {1, 2, 3, 7, 32, 64};
  /* prompts ending at 31/32/33, 255/256/257 and 513 positions, and warm contexts */
  const int ends[] = {31, 32, 33, 255, 256, 257, 513}, warm[] = {1, 33, 256, 300};
  for (size_t s = 0; s < sizeof(shapes) / sizeof(*shapes); s++)
    for (size_t mi = 0; mi < sizeof(ms) / sizeof(*ms); mi++) {
      for (size_t e = 0; e < sizeof(ends) / sizeof(*ends); e++)
        if (ends[e] - ms[mi] >= 0)
          attention_batch_head_case(shapes[s], ms[mi], ends[e] - ms[mi]);
      for (size_t w = 0; w < sizeof(warm) / sizeof(*warm); w++)
        attention_batch_head_case(shapes[s], ms[mi], warm[w]);
    }
  if (with_forced) { /* the multi-chunk plans of many-worker sockets, every position near a chunk edge */
    const int workers[] = {3, 6, 56};
    for (size_t w = 0; w < sizeof(workers) / sizeof(*workers); w++) {
      attn_force_workers = workers[w];
      for (int pos0 = 0; pos0 <= 70; pos0 += (pos0 < 40 ? 1 : 3)) {
        attention_batch_head_case((AttnShape){2, 4, 64}, 7, pos0);
        attention_batch_head_case((AttnShape){2, 1, 40}, 9, pos0);
      }
      attention_batch_head_case((AttnShape){2, 2, 64}, 33, 100);
      attention_batch_head_case((AttnShape){2, 8, 64}, 5, 250);
    }
    attn_force_workers = 0;
  }
  attention_batch_head_case((AttnShape){2, 4, 128}, 64, 1023);
}

/* LLAMA_ATTN=amx (prefill-gap fix 7): the AMX-BF16 flash attention against the split kernel on the head-major layout,
 * within a BF16 budget (Q, K, P and V are rounded to BF16; values in [-1, 1]). It runs wherever AMX can be enabled: the
 * hardware tiles, or their software model in test-kernels-emul, with the AVX-512 softmax at the avx512/amx levels and
 * the portable one at the scalar level. Shapes cover every query-head group up to 32 (one row per task), head sizes
 * 32..256, microbatches that end mid-block and warm contexts that start mid-block.
 * It is also partition-invariant, bitwise: each KV head run as its own call (one KV head, its query heads: a socket's
 * shard under tensor parallelism) gives that head's slice of the all-heads result, and a team of any size gives the
 * serial result (team != NULL: the caller's team, restored afterwards). Key blocks start at position 0 whatever the
 * call, and each output row is one task's (KV head x query block), so neither the split nor the thread count moves a
 * rounding. */
static double worst_amx;

static void amx_attention_run(float *out, const float *q, const float *kc, const float *vc, float *att, int heads,
                              int kv, int hd, int seq, int pos0, int m) {
  AttentionWork work;
  k_attention_init_rows(&work, heads, kv, hd, seq, m, NULL);
  k_attention_batch_layout(out, q, kc, vc, att, &work, heads, kv, hd, seq, pos0, m, (size_t)hd, (size_t)seq * hd);
  k_attention_free(&work);
}

static void test_attention_amx(struct Team *team) {
  if (gemm_amx_unavailable())
    return;
  int was = gemm_amx_active();
  gemm_amx_enable(1);
  const AttnShape shapes[] = {{2, 1, 64}, {2, 4, 128}, {2, 8, 64}, {1, 2, 32}, {1, 32, 32}, {1, 4, 256}};
  const int ms[] = {2, 7, 33, 64}, starts[] = {0, 1, 31, 100};
  for (size_t s = 0; s < sizeof(shapes) / sizeof(*shapes); s++)
    for (size_t mi = 0; mi < sizeof(ms) / sizeof(*ms); mi++)
      for (size_t pi = 0; pi < sizeof(starts) / sizeof(*starts); pi++) {
        const int kv = shapes[s].kv_heads, heads = kv * shapes[s].group, hd = shapes[s].hd, m = ms[mi];
        const int pos0 = starts[pi], seq = pos0 + m + 5, dim = heads * hd;
        float *q = zeroed((size_t)m * dim, 4), *kc = zeroed((size_t)kv * seq * hd, 4),
              *vc = zeroed((size_t)kv * seq * hd, 4), *want = zeroed((size_t)m * dim, 4),
              *got = zeroed((size_t)m * dim, 4), *att = zeroed((size_t)heads * seq, 4);
        fill(q, (size_t)m * dim, 1.0f);
        fill(kc, (size_t)kv * seq * hd, 1.0f);
        fill(vc, (size_t)kv * seq * hd, 1.0f);
        for (int variant = 0; variant < 2; variant++) {
          assert(kernels_attention_select(variant ? "amx" : "split", NULL));
          AttentionWork work;
          k_attention_init_rows(&work, heads, kv, hd, seq, m, NULL);
          checks++;
          if (variant && !work.kv_bf16 && failures++ < 20)
            fprintf(stderr, "attention amx [%s]: no BF16 K/V workspace\n", current);
          k_attention_batch_layout(variant ? got : want, q, kc, vc, att, &work, heads, kv, hd, seq, pos0, m, (size_t)hd,
                                   (size_t)seq * hd);
          k_attention_free(&work);
        }
        /* partition invariance: per KV head (a socket's shard), and serial against the team */
        const int group = shapes[s].group, ldl = group * hd;
        float *qs = zeroed((size_t)m * ldl, 4), *part = zeroed((size_t)m * dim, 4);
        assert(kernels_attention_select("amx", NULL));
        for (int g = 0; kv > 1 && g < kv; g++) {
          for (int r = 0; r < m; r++)
            memcpy(qs + (size_t)r * ldl, q + (size_t)r * dim + (size_t)g * ldl, (size_t)ldl * sizeof(float));
          amx_attention_run(part, qs, kc + (size_t)g * seq * hd, vc + (size_t)g * seq * hd, att, group, 1, hd, seq,
                            pos0, m);
          int same = 1;
          for (int r = 0; r < m; r++)
            same &=
                !memcmp(part + (size_t)r * ldl, got + (size_t)r * dim + (size_t)g * ldl, (size_t)ldl * sizeof(float));
          checks++;
          if (!same && failures++ < 20)
            fprintf(stderr,
                    "attention amx [%s]: KV head %d alone differs from the all-heads call (kv=%d group=%d hd=%d m=%d "
                    "pos0=%d)\n",
                    current, g, kv, group, hd, m, pos0);
        }
        if (team) {
          kernels_set_team(NULL);
          amx_attention_run(part, q, kc, vc, att, heads, kv, hd, seq, pos0, m);
          kernels_set_team(team);
          checks++;
          if (memcmp(part, got, (size_t)m * dim * sizeof(float)) && failures++ < 20)
            fprintf(stderr,
                    "attention amx [%s]: the team differs from the serial run (kv=%d group=%d hd=%d m=%d pos0=%d)\n",
                    current, kv, group, hd, m, pos0);
        }
        free(qs), free(part);
        assert(kernels_attention_select("split", NULL));
        double worst = 0;
        for (size_t i = 0; i < (size_t)m * dim; i++) {
          double d = fabs((double)got[i] - want[i]);
          worst = isnan(d) ? INFINITY : d > worst ? d : worst;
        }
        worst_amx = worst > worst_amx ? worst : worst_amx;
        char what[96];
        snprintf(what, sizeof(what), "attention amx kv=%d group=%d hd=%d m=%d pos0=%d", kv, shapes[s].group, hd, m,
                 pos0);
        check(what, worst, 0.0, 0.03);
        free(q), free(kc), free(vc), free(want), free(got), free(att);
      }
  gemm_amx_enable(was);
}

#ifndef __has_feature
#define __has_feature(x) 0
#endif

static double matrix_seconds; /* total time of the matrix, printed at the end so its cost stays visible */

static void test_attention_matrix(void) {
  double started = monotonic_seconds();
  /* Reduced head counts (two KV heads) keep the matrix fast on scalar and sanitizer builds; the group sizes, head
   * sizes and position counts are the ones that matter for the chunk plan and the tail masks. */
  const int groups[] = {1, 2, 4, 8}, head_sizes[] = {64, 128, 40}, ms[] = {1, 3, 7, 32};
  const int ends[] = {31, 32, 33, 255, 256, 257}, warm[] = {1, 33, 256};
  const int decode_positions[] = {0, 1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 288, 511, 512, 513};
  int reduced = strstr(current, "teams") != NULL; /* the team loop repeats the matrix per team size: fewer shapes */
#if defined(__SANITIZE_THREAD__) || __has_feature(thread_sanitizer)
  int sparse = reduced; /* ThreadSanitizer makes the team barriers slow: one shape per team size is enough there */
#else
  int sparse = 0;
#endif
  for (size_t g = 0; g < sizeof(groups) / sizeof(*groups); g++)
    for (size_t h = 0; h < sizeof(head_sizes) / sizeof(*head_sizes); h++) {
      AttnShape shape = {.kv_heads = 2, .group = groups[g], .hd = head_sizes[h]};
      if (reduced && (groups[g] == 4 || head_sizes[h] == 128))
        continue;
      if (sparse && (groups[g] != 2 || head_sizes[h] != 40))
        continue;
      for (size_t mi = 0; mi < sizeof(ms) / sizeof(*ms); mi++) {
        /* prompts that end (last row included) at 31/32/33 and 255/256/257 positions, and warm contexts */
        for (size_t e = 0; e < sizeof(ends) / sizeof(*ends); e++)
          if (ends[e] - ms[mi] >= 0 && !(reduced && ends[e] != 32 && ends[e] != 257))
            attention_batch_case(shape, ms[mi], ends[e] - ms[mi], 0);
        for (size_t w = 0; w < sizeof(warm) / sizeof(*warm); w++)
          if (!reduced || warm[w] == 1)
            attention_batch_case(shape, ms[mi], warm[w], 0);
      }
      attention_decode_case(shape, decode_positions, sizeof(decode_positions) / sizeof(*decode_positions));
    }
  /* M = 256: a microbatch that spans several chunk plans. Prompts ending at 256/257 positions and warm contexts (the
   * reference is O(m * context), so each shape gets two of them), then the same wave split from the workspace budget
   * the engine sizes (k_attention_set_workspace_mib) instead of a hand cut. */
  if (!sparse) {
    attention_batch_case((AttnShape){2, 1, 64}, 256, 0, 0);
    attention_batch_case((AttnShape){2, 4, 64}, 256, 1, 0);
  }
  if (!reduced) {
    attention_batch_case((AttnShape){2, 8, 40}, 256, 0, 0);
    attention_batch_case((AttnShape){2, 8, 40}, 256, 33, 0);
    attention_batch_case((AttnShape){2, 8, 40}, 256, 33, 1);
    attention_batch_case((AttnShape){2, 2, 128}, 256, 1, 0);
    attention_batch_case((AttnShape){2, 2, 128}, 256, 256, 0);
  }
  matrix_seconds += monotonic_seconds() - started;
}

int main(int argc, char **argv) {
  if (argc >= 2 && !strcmp(argv[1], "bench-attention")) {
    kernels_init();
    bench_attention(argc > 2 ? atoi(argv[2]) : 4);
    return 0;
  }
  enum { ROPE_FLOATS = 1 << 16, Q8_VALUES = 1 << 14 };
  float *rope[3] = {zeroed(ROPE_FLOATS, sizeof(float)), zeroed(ROPE_FLOATS, sizeof(float)),
                    zeroed(ROPE_FLOATS, sizeof(float))};
  int8_t *quantized[3] = {zeroed(Q8_VALUES, 1), zeroed(Q8_VALUES, 1), zeroed(Q8_VALUES, 1)};
  float *scales[3] = {zeroed(Q8_VALUES, sizeof(float)), zeroed(Q8_VALUES, sizeof(float)),
                      zeroed(Q8_VALUES, sizeof(float))};
  size_t rope_used = 0;
  int ran[3] = {0, 0, 0};
  /* The amx level runs every kernel on its AVX-512 path (AMX only touches the batched GEMM tiles), so this repeats the
   * whole suite under LLAMA_KERNELS=amx; it is skipped where AMX or its permission is unavailable (C4). */
  const KernelIsa isas[3] = {KERNELS_SCALAR, KERNELS_AVX512, KERNELS_AMX};
  const char *const isa_names[3] = {"scalar", "avx512", "amx"};
  for (int i = 0; i < 3; i++) {
    unsigned long long seed = rng;
    if (!kernels_select(isas[i])) {
      printf("kernels: %s unavailable on this CPU, skipped\n", isa_names[i]);
      continue;
    }
    current = kernels_name();
    int before = failures;
    test_f32();
    test_q8(quantized[i], scales[i]);
    test_norm();
    rope_used = test_rope(rope[i]);
    test_attention();
    test_attention_batch();
    test_attention_batch_waves();
    test_attention_matrix();
    test_attention_decode_long();
    test_attention_head_layout(1);
    test_attention_batch_head_layout(1);
    test_attention_forced_plans();
    test_attention_empty_partial();
    test_attention_tiled_hook();
    test_partial();
    test_row_parallel();
    test_two_teams();
    test_attention_split();
    test_attention_amx(NULL);
    rng = seed; /* identical inputs for every instruction set */
    ran[i] = 1;
    printf("kernels: %s %s\n", current, failures == before ? "ok" : "FAILED");
  }
  for (int other = 1; other < 3; other++) {
    if (!(ran[0] && ran[other]))
      continue;
    current = other == 1 ? "scalar vs avx512" : "scalar vs amx";
    checks += 3;
    if (memcmp(rope[0], rope[other], rope_used * sizeof(float)) && failures++ < 20)
      fprintf(stderr, "rope [%s]: results differ bitwise\n", current);
    if (memcmp(quantized[0], quantized[other], Q8_VALUES) && failures++ < 20)
      fprintf(stderr, "quantize [%s]: int8 values differ bitwise\n", current);
    if (memcmp(scales[0], scales[other], Q8_VALUES * sizeof(float)) && failures++ < 20)
      fprintf(stderr, "quantize [%s]: scales differ bitwise\n", current);
  }
  /* The same kernels on persistent worker teams. Odd sizes stress the range splitting, and every size runs
   * against the same double-precision references as the serial path. */
  int team_sizes[8] = {1, 2, 3, 5, 7};
  size_t team_size_count = 5;
  const char *override = getenv("LLAMA_TEST_TEAMS"); /* e.g. "1,56" for a full socket on the cluster */
  if (override && *override) {
    team_size_count = 0;
    for (char *copy = strdup(override), *token = strtok(copy, ","); token && team_size_count < 8;
         token = strtok(NULL, ","))
      team_sizes[team_size_count++] = atoi(token);
  }
  int8_t *scratch_q = zeroed(Q8_VALUES, 1);
  float *scratch_s = zeroed(Q8_VALUES, sizeof(float));
  static char label[64];
  for (int i = 0; i < 3; i++) {
    if (!ran[i])
      continue;
    kernels_select(isas[i]);
    for (size_t t = 0; t < team_size_count; t++) {
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
      test_attention_batch();
      test_attention_matrix();
      test_attention_head_layout(0);
      test_attention_batch_head_layout(0);
      test_attention_amx(pool_team(0));
      test_partial();
      test_gemv_stagger();
      test_gemv_rows();
      test_row_parallel();
      k_gemv_force_hybrid(1); /* the hybrid split of large FP32 jobs, on these small matrices */
      test_f32();
      test_partial();
      test_row_parallel();
      k_gemv_force_hybrid(0);
      printf("kernels: %s %s\n", current, failures == before ? "ok" : "FAILED");
      kernels_set_team(NULL);
      pool_stop();
    }
  }
  free(scratch_q), free(scratch_s);
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
  for (int i = 0; i < 3; i++)
    free(rope[i]), free(quantized[i]), free(scales[i]);
  printf("kernels: attention matrix %.1f s, long decode contexts %.1f s\n", matrix_seconds, matrix_seconds_long);
  printf("kernels: tiled attention, largest error %.3g against the double reference, %.3g against split\n",
         worst_vs_reference, worst_vs_split);
  printf("kernels: AMX-BF16 attention, largest error %.3g against split\n", worst_amx);
  printf("kernels: %d checks, %d failures\n", checks, failures);
  return failures != 0;
}
