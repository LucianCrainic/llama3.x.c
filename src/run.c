/* LLaMA 3 CPU inference, derived from Andrej Karpathy and James W. DeLancey.
 * See LICENSE. Numerical kernels intentionally remain independent per precision. */
#include "runtime.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef struct {
  // token embedding table
  const float *token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  const float *rms_att_weight; // (layer, dim) rmsnorm weights
  const float *rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  const float *wq; // (layer, dim, n_heads * head_size)
  const float *wk; // (layer, dim, n_kv_heads * head_size)
  const float *wv; // (layer, dim, n_kv_heads * head_size)
  const float *wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  const float *w1; // (layer, hidden_dim, dim)
  const float *w2; // (layer, dim, hidden_dim)
  const float *w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  const float *rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  const float *wcls;
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x;      // activation at current time stamp (dim,)
  float *xb;     // same, but inside a residual branch (dim,)
  float *xb2;    // an additional buffer just for convenience (dim,)
  float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
  float *q;      // query (dim,)
  float *k;      // key (dim,)
  float *v;      // value (dim,)
  float *att;    // buffer for scores/attention values (n_heads, seq_len)
  float *logits; // output logits
  // kv cache
  float *key_cache;   // (layer, seq_len, kv_dim); cache persists for this sequence
  float *value_cache; // (layer, seq_len, kv_dim); cache persists for this sequence
} RunState;

typedef struct {
  Config config;              // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state;             // buffers for the "wave" of activations in the forward pass
  // Mapping ownership: weights borrow this region until engine_free().
  int fd;           // file descriptor for memory mapping
  float *data;      // memory mapped data pointer
  size_t file_size; // size of the checkpoint file in bytes
} Transformer;

static void malloc_run_state(RunState *s, Config *p) {
  // Zeroed activations and cache are owned for the lifetime of this model.
  int kv_dim = (p->dim / p->n_heads) * p->n_kv_heads;
  s->x = memory_alloc(p->dim, sizeof(float), MEM_ACTIVATION);
  s->xb = memory_alloc(p->dim, sizeof(float), MEM_ACTIVATION);
  s->xb2 = memory_alloc(p->dim, sizeof(float), MEM_ACTIVATION);
  s->hb = memory_alloc(p->hidden_dim, sizeof(float), MEM_ACTIVATION);
  s->hb2 = memory_alloc(p->hidden_dim, sizeof(float), MEM_ACTIVATION);
  s->q = memory_alloc(p->dim, sizeof(float), MEM_ACTIVATION);
  s->key_cache = memory_alloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float), MEM_KV);
  s->value_cache = memory_alloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float), MEM_KV);
  s->att = memory_alloc((size_t)p->n_heads * p->seq_len, sizeof(float), MEM_ACTIVATION);
  s->logits = memory_alloc(p->vocab_size, sizeof(float), MEM_ACTIVATION);
}

static void free_run_state(RunState *s) {
  memory_free(s->x);
  memory_free(s->xb);
  memory_free(s->xb2);
  memory_free(s->hb);
  memory_free(s->hb2);
  memory_free(s->q);
  memory_free(s->att);
  memory_free(s->logits);
  memory_free(s->key_cache);
  memory_free(s->value_cache);
}

static void memory_map_weights(TransformerWeights *w, const Config *p, const float *ptr, int shared_weights) {
  int head_size = p->dim / p->n_heads;
  // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
  size_t n_layers = (size_t)p->n_layers;
  w->token_embedding_table = ptr;
  ptr += (size_t)p->vocab_size * p->dim;
  w->rms_att_weight = ptr;
  ptr += n_layers * p->dim;
  w->wq = ptr;
  ptr += n_layers * (size_t)p->dim * (p->n_heads * head_size);
  w->wk = ptr;
  ptr += n_layers * (size_t)p->dim * (p->n_kv_heads * head_size);
  w->wv = ptr;
  ptr += n_layers * (size_t)p->dim * (p->n_kv_heads * head_size);
  w->wo = ptr;
  ptr += n_layers * (size_t)(p->n_heads * head_size) * p->dim;
  w->rms_ffn_weight = ptr;
  ptr += n_layers * p->dim;
  w->w1 = ptr;
  ptr += n_layers * (size_t)p->dim * p->hidden_dim;
  w->w2 = ptr;
  ptr += n_layers * (size_t)p->hidden_dim * p->dim;
  w->w3 = ptr;
  ptr += n_layers * (size_t)p->dim * p->hidden_dim;
  w->rms_final_weight = ptr;
  ptr += p->dim;
  ptr += (size_t)p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
  ptr += (size_t)p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
  w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

static void read_checkpoint(const char *path, Config *config, TransformerWeights *weights, int *fd, float **data,
                            size_t *file_size) {
  int shared, gs;
  double start = monotonic_seconds();
  checkpoint_map(path, 0, config, &shared, &gs, fd, data, file_size);
  memory_map_weights(weights, config, (void *)((char *)*data + 28), shared);
  metrics.checkpoint_s = monotonic_seconds() - start;
  metrics.checkpoint_ready = 1;
}

static void free_transformer(Transformer *t) {
  // close the memory mapping
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
  // free the RunState buffers
  free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

static void matmul(float *xout, const float *x, const float *w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[(size_t)i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

/* One autoregressive position: embeddings -> attention/RoPE over cached prefix
 * -> SwiGLU FFN per layer -> normalized vocabulary logits. Cache rows are
 * [layer][position][KV head * head_size]; GQA query heads share KV heads. */
static float *forward(Transformer *transformer, int token, int pos) {

  // a few convenience variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim / p->n_heads) * p->n_kv_heads;
  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;

  // copy the token embedding into x
  const float *content_row = w->token_embedding_table + (size_t)token * dim;
  memcpy(x, content_row, dim * sizeof(*x));

  // forward all the layers
  for (size_t l = 0; l < (size_t)p->n_layers; l++) {

    // attention rmsnorm
    rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

    // key and value point to the kv cache
    size_t loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + (size_t)pos * kv_dim;
    s->v = s->value_cache + loff + (size_t)pos * kv_dim;

    // qkv matmuls for this position
    matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
    matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
    matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    for (int i = 0; i < p->n_heads; i++) {
      for (int j = 0; j < head_size; j += 2) {
        float freq = 1.0f / powf(ROPE_THETA, (float)j / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        float q0 = s->q[i * head_size + j];
        float q1 = s->q[i * head_size + j + 1];
        s->q[i * head_size + j] = q0 * fcr - q1 * fci;
        s->q[i * head_size + j + 1] = q0 * fci + q1 * fcr;
        if (i < p->n_kv_heads) {
          float k0 = s->k[i * head_size + j];
          float k1 = s->k[i * head_size + j + 1];
          s->k[i * head_size + j] = k0 * fcr - k1 * fci;
          s->k[i * head_size + j + 1] = k0 * fci + k1 * fcr;
        }
      }
    }

    // multihead attention. iterate over all heads
    int h;
#ifdef _OPENMP
#pragma omp parallel for private(h)
#endif
    for (h = 0; h < p->n_heads; h++) {
      // get the query vector for this head
      float *q = s->q + h * head_size;
      // attention scores for this head
      float *att = s->att + (size_t)h * p->seq_len;
      // iterate over all timesteps, including the current one
      for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        float *k = s->key_cache + loff + (size_t)t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
          score += q[i] * k[i];
        }
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax(att, pos + 1);

      // weighted sum of the values, store back into xb
      float *xb = s->xb + h * head_size;
      memset(xb, 0, head_size * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        float *v = s->value_cache + loff + (size_t)t * kv_dim + (h / kv_mul) * head_size;
        // get the attention weight for this timestep
        float a = att[t];
        // accumulate the weighted value into xb
        for (int i = 0; i < head_size; i++) {
          xb[i] += a * v[i];
        }
      }
    }

    // final matmul to get the output of the attention
    matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

    // residual connection back into x
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb2[i];
    }

    // ffn rmsnorm
    rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
    matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

    // SwiGLU non-linearity
    for (int i = 0; i < hidden_dim; i++) {
      float val = s->hb[i];
      // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
      val *= (1.0f / (1.0f + expf(-val)));
      // elementwise multiply with w3(x)
      val *= s->hb2[i];
      s->hb[i] = val;
    }

    // final matmul to get the output of the ffn
    matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

    // residual connection
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb[i];
    }
  }

  // final rmsnorm
  rmsnorm(x, x, w->rms_final_weight, dim);

  // classifier into logits
  matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
  return s->logits;
}

/* CLI adapter: each executable links exactly one numerical implementation. */
void engine_load(Model *model, const char *path) {
  Transformer *t = memory_alloc(1, sizeof(*t), MEM_METADATA);
  t->fd = -1;
  model->implementation = t;
  read_checkpoint(path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
  model->config = t->config;
  model->mapped_bytes = t->file_size;
  double start = monotonic_seconds();
  malloc_run_state(&t->state, &t->config);
  metrics.runtime_s = monotonic_seconds() - start;
  metrics.runtime_ready = 1;
}
float *engine_forward(Model *model, int token, int position) {
  if (token < 0 || token >= model->config.vocab_size || position < 0 || position >= model->config.seq_len)
    fail("forward input out of range: token=%d position=%d", token, position);
  return forward(model->implementation, token, position);
}
void engine_free(Model *model) {
  if (model->implementation) {
    free_transformer(model->implementation);
    memory_free(model->implementation);
    model->implementation = NULL;
  }
}
const char *engine_precision(void) {
  return "fp32";
}
int engine_group_size(void) {
  return 0;
}
