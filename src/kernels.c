#include "runtime.h"
#include <math.h>

void rmsnorm(float *o, const float *x, const float *weight, int size) {
  rmsnorm_eps(o, x, weight, size, 1e-5f);
}

void rmsnorm_eps(float *o, const float *x, const float *weight, int size, float epsilon) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += epsilon;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

void softmax(float *x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  if (!isfinite(max_val))
    fail("nonfinite value in softmax");
  for (int i = 1; i < size; i++) {
    if (!isfinite(x[i]))
      fail("nonfinite value in softmax");
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

/* Llama 3.1/3.2 wavelength-dependent scaling. Q/K rows use Meta's adjacent
 * complex pairs; HF exporters must undo the split-half Q/K permutation. */
float rope_frequency(const Config *p, int j) {
  float freq = 1.0f / powf(p->rope_theta, (float)j / (float)p->head_dim);
  if (p->rope_type == 1) {
    float wavelength = 6.2831853071795864769f / freq;
    if (wavelength > p->rope_original / p->rope_low)
      freq /= p->rope_factor;
    else if (wavelength >= p->rope_original / p->rope_high) {
      float smooth = (p->rope_original / wavelength - p->rope_low) / (p->rope_high - p->rope_low);
      freq = (1 - smooth) * freq / p->rope_factor + smooth * freq;
    }
  }
  return freq;
}
void apply_rope(float *q, float *k, const Config *p, int position) {
  for (int h = 0; h < p->n_heads; h++) {
    for (int j = 0; j < p->head_dim; j += 2) {
      float angle = position * rope_frequency(p, j), c = cosf(angle), s = sinf(angle);
      int i = h * p->head_dim + j;
      float a = q[i], b = q[i + 1];
      q[i] = a * c - b * s;
      q[i + 1] = a * s + b * c;
      if (h < p->n_kv_heads) {
        a = k[i];
        b = k[i + 1];
        k[i] = a * c - b * s;
        k[i + 1] = a * s + b * c;
      }
    }
  }
}
