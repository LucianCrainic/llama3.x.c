#include "runtime.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef LLAMA_PCRE2
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#endif

/* Same ordered alternatives as Meta's Llama 3 tokenizer. UCP supplies Unicode
 * letter/number/whitespace properties; UTF rejects malformed prompt bytes. */
#ifdef LLAMA_PCRE2
static const char llama_pattern[] = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|"
                                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
#endif

static int compare_tokens(const void *a, const void *b) {
  return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, const char *tokenizer_path, int vocab_size) {
  t->vocab_size = vocab_size;
  t->vocab = memory_alloc(vocab_size, sizeof(char *), MEM_TOKENIZER);
  t->vocab_scores = memory_alloc(vocab_size, sizeof(float), MEM_TOKENIZER);
  t->sorted_vocab = NULL;
  t->lengths = memory_alloc(vocab_size, sizeof(int), MEM_TOKENIZER);
  t->bos_id = BOS_ID;
  t->eos_id = EOS_ID;
  t->eot_id = EOT_ID;
  t->eom_id = -1;
  t->start_header_id = 128006;
  t->end_header_id = 128007;
  memset(t->byte_pieces, 0, sizeof(t->byte_pieces));
  for (int i = 0; i < 256; i++)
    t->byte_pieces[i * 2] = (unsigned char)i;
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file)
    fail("cannot open tokenizer '%s': %s", tokenizer_path, strerror(errno));
  if (fread(&t->max_token_length, 4, 1, file) != 1 || !t->max_token_length ||
      (t->max_token_length > 1048576 && t->max_token_length != 0x336b6f74))
    fail("invalid tokenizer header in '%s' (maximum token length must be 1..1048576)", tokenizer_path);
  if (t->max_token_length == 0x336b6f74) { /* tok3, tokenizer format 1 */
    int h[11];
    if (fread(h, 4, 11, file) != 11 || (h[0] != 1 && h[0] != 2) || h[1] != vocab_size || h[2] < 256 ||
        h[2] > vocab_size || h[3] <= 0 || h[3] > 1048576)
      fail("invalid versioned tokenizer header");
    t->version = h[0];
    t->base_tokens = h[2];
    t->max_token_length = (unsigned)h[3];
    t->bos_id = h[4];
    t->eos_id = h[5];
    t->eot_id = h[6];
    t->eom_id = h[7];
    t->start_header_id = h[8];
    t->end_header_id = h[9];
    if ((h[0] == 1 && h[10]) || h[10] < 0 || h[10] > 1048576)
      fail("unsupported tokenizer prefix length");
    t->chat_prefix = memory_alloc((size_t)h[10] + 1, 1, MEM_TOKENIZER);
    if (fread(t->chat_prefix, 1, (size_t)h[10], file) != (size_t)h[10] || memchr(t->chat_prefix, 0, (size_t)h[10]))
      fail("invalid tokenizer system prefix");
    int ids[] = {h[4], h[5], h[6], h[7], h[8], h[9]};
    for (int i = 0; i < 6; i++)
      if (ids[i] < (i < 2 ? 0 : -1) || ids[i] >= vocab_size || (ids[i] >= 0 && ids[i] < t->base_tokens))
        fail("invalid tokenizer special token ID");
#ifdef LLAMA_PCRE2
    int error;
    PCRE2_SIZE offset;
    t->regex =
        pcre2_compile((PCRE2_SPTR)llama_pattern, PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_UCP, &error, &offset, NULL);
    if (!t->regex)
      fail("cannot compile tokenizer regex at %zu (error %d)", (size_t)offset, error);
#else
    fail("versioned tokenizers require a PCRE2=1 build");
#endif
  }
  for (int i = 0; i < vocab_size; i++) {
    int len;
    if (fread(t->vocab_scores + i, 4, 1, file) != 1 || fread(&len, 4, 1, file) != 1 || len <= 0 ||
        (unsigned)len > t->max_token_length || !isfinite(t->vocab_scores[i]))
      fail("invalid/truncated tokenizer '%s' at token %d", tokenizer_path, i);
    if (t->version && t->vocab_scores[i] != (float)i)
      fail("tokenizer rank does not match ID %d", i);
    t->lengths[i] = len;
    t->vocab[i] = memory_alloc((size_t)len + 1, 1, MEM_TOKENIZER);
    if (fread(t->vocab[i], 1, (size_t)len, file) != (size_t)len)
      fail("truncated tokenizer '%s' at token %d", tokenizer_path, i);
  }
  if (fgetc(file) != EOF)
    fail("tokenizer '%s' has extra entries: vocabulary does not match model", tokenizer_path);
  if (ferror(file))
    fail("error reading tokenizer '%s'", tokenizer_path);
  fclose(file);
}

void free_tokenizer(Tokenizer *t) {
  for (int i = 0; i < t->vocab_size; i++) {
    memory_free(t->vocab[i]);
  }
  memory_free(t->vocab);
  memory_free(t->vocab_scores);
  memory_free(t->sorted_vocab);
  memory_free(t->lengths);
  memory_free(t->chat_prefix);
#ifdef LLAMA_PCRE2
  pcre2_code_free(t->regex);
#endif
}

const char *decode(Tokenizer *t, int token) {
  if (token < 0 || token >= t->vocab_size)
    fail("invalid decoded token %d", token);
  char *piece = t->vocab[token];

  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (!t->version && sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void print_token(Tokenizer *t, int token) {
  const char *piece = decode(t, token);
  if (t->version) {
    size_t n = (size_t)t->lengths[token];
    if (fwrite(piece, 1, n, stdout) != n)
      fail("cannot write decoded token");
  } else
    safe_printf(piece);
}

void safe_printf(const char *piece) {
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  // efficiently find the perfect match for str in vocab, return its index or -1 if not found
  TokenIndex tok = {.str = str}; // acts as the key to search for
  TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

/* Binary lookup is length-aware: byte zero is a valid rank-0..255 token. */
#ifdef LLAMA_PCRE2
static int compare_bytes(const void *a, const void *b) {
  const TokenIndex *x = a, *y = b;
  size_t n = x->length < y->length ? x->length : y->length;
  int c = memcmp(x->str, y->str, n);
  return c ? c : (x->length > y->length) - (x->length < y->length);
}
static int rank_lookup(Tokenizer *t, const char *bytes, size_t length) {
  TokenIndex key = {.str = (char *)bytes, .length = length};
  TokenIndex *r = bsearch(&key, t->sorted_vocab, t->base_tokens, sizeof(key), compare_bytes);
  return r ? r->id : -1;
}
static void prepare_ranks(Tokenizer *t) {
  if (t->sorted_vocab)
    return;
  t->sorted_vocab = memory_alloc(t->base_tokens, sizeof(TokenIndex), MEM_TOKENIZER);
  for (int i = 0; i < t->base_tokens; i++)
    t->sorted_vocab[i] = (TokenIndex){t->vocab[i], i, (size_t)t->lengths[i]};
  qsort(t->sorted_vocab, t->base_tokens, sizeof(TokenIndex), compare_bytes);
  for (int i = 1; i < t->base_tokens; i++)
    if (!compare_bytes(t->sorted_vocab + i - 1, t->sorted_vocab + i))
      fail("duplicate tokenizer byte piece");
  for (int i = 0; i < 256; i++) {
    char byte = (char)i;
    if (rank_lookup(t, &byte, 1) < 0)
      fail("tokenizer missing byte %d", i);
  }
}
static void encode_ranked(Tokenizer *t, const char *text, int bos, int eos, int *tokens, int *count) {
  prepare_ranks(t);
  size_t length = strlen(text), offset = 0;
  *count = 0;
  if (bos)
    tokens[(*count)++] = t->bos_id;
  pcre2_match_data *match = pcre2_match_data_create_from_pattern(t->regex, NULL);
  if (!match)
    fail("tokenizer regex allocation failed");
  while (offset < length) {
    int rc = pcre2_match(t->regex, (PCRE2_SPTR)text, length, offset, PCRE2_ANCHORED, match, NULL);
    if (rc < 1)
      fail("invalid UTF-8 or tokenizer regex error %d at byte %zu", rc, offset);
    PCRE2_SIZE *span = pcre2_get_ovector_pointer(match);
    size_t end = span[1], n = end - offset;
    if (!n)
      fail("empty tokenizer regex match");
    int direct = rank_lookup(t, text + offset, n);
    if (direct >= 0)
      tokens[(*count)++] = direct;
    else {
      /* Merge the lowest-ranked adjacent pair, leftmost on ties. Never merge
       * across regex pieces or use special token spellings as ordinary ranks. */
      size_t *bounds = memory_alloc(n + 1, sizeof(size_t), MEM_PROMPT);
      for (size_t i = 0; i <= n; i++)
        bounds[i] = i;
      size_t parts = n;
      while (parts > 1) {
        int best = INT_MAX;
        size_t at = 0;
        for (size_t i = 0; i + 1 < parts; i++) {
          int rank = rank_lookup(t, text + offset + bounds[i], bounds[i + 2] - bounds[i]);
          if (rank >= 0 && rank < best) {
            best = rank;
            at = i;
          }
        }
        if (best == INT_MAX)
          break;
        memmove(bounds + at + 1, bounds + at + 2, (parts - at - 1) * sizeof(size_t));
        parts--;
      }
      for (size_t i = 0; i < parts; i++) {
        int rank = rank_lookup(t, text + offset + bounds[i], bounds[i + 1] - bounds[i]);
        if (rank < 0)
          fail("tokenizer byte fallback missing");
        tokens[(*count)++] = rank;
      }
      memory_free(bounds);
    }
    offset = end;
  }
  pcre2_match_data_free(match);
  if (eos)
    tokens[(*count)++] = t->eos_id;
}
#endif

void tokenizer_check_model(const Tokenizer *t, const Config *p) {
  if (p->format == 3 && (!t->version || t->bos_id != p->bos_id || t->eos_id != p->eos_id || t->eot_id != p->eot_id ||
                         t->eom_id != p->eom_id))
    fail("version 3 model requires a versioned tokenizer with matching special IDs");
}
int tokenizer_stop(const Tokenizer *t, int token) {
  return token == t->eos_id || token == t->eot_id || token == t->eom_id;
}

/* Python str.strip() whitespace set used by the official basic formatter.
 * Decode only the edge codepoint; malformed UTF-8 still fails in encode(). */
static size_t trim_space(const char *s, size_t n) {
  if (!n)
    return 0;
  unsigned char b = (unsigned char)s[0];
  if ((b >= 9 && b <= 13) || (b >= 28 && b <= 32))
    return 1;
  if (n >= 2 && b == 0xc2 && ((unsigned char)s[1] == 0x85 || (unsigned char)s[1] == 0xa0))
    return 2;
  if (n >= 3 && b == 0xe1 && (unsigned char)s[1] == 0x9a && (unsigned char)s[2] == 0x80)
    return 3;
  if (n >= 3 && b == 0xe2) {
    unsigned char c = (unsigned char)s[1], d = (unsigned char)s[2];
    if ((c == 0x80 && ((d >= 0x80 && d <= 0x8a) || d == 0xa8 || d == 0xa9 || d == 0xaf)) || (c == 0x81 && d == 0x9f))
      return 3;
  }
  if (n >= 3 && b == 0xe3 && (unsigned char)s[1] == 0x80 && (unsigned char)s[2] == 0x80)
    return 3;
  return 0;
}

/* Ordinary content is encoded without recognizing special spellings. Only this
 * template injects special IDs. Empty system prompts omit the system message. */
int *chat_prompt(Tokenizer *t, const char *system, const char *user, int first, int *count) {
  if (t->start_header_id < 0 || t->end_header_id < 0 || t->eot_id < 0 || t->vocab_size <= t->end_header_id)
    fail("tokenizer lacks Llama chat header metadata");
  const char *prefix = t->chat_prefix ? t->chat_prefix : "";
  size_t capacity = checked_add(checked_add(checked_add(strlen(system), strlen(user)), strlen(prefix)), 128);
  if (capacity > INT_MAX)
    fail("chat prompt is too long");
  int *tokens = memory_alloc(capacity, sizeof(int), MEM_PROMPT);
  int n = 0, k;
  if (first)
    tokens[n++] = t->bos_id;
  const char *roles[] = {"system", "user", "assistant"};
  const char *contents[] = {system, user, ""};
  for (int i = first && (*system || *prefix) ? 0 : 1; i < 3; i++) {
    tokens[n++] = t->start_header_id;
    encode(t, roles[i], 0, 0, tokens + n, &k);
    n += k;
    tokens[n++] = t->end_header_id;
    encode(t, "\n\n", 0, 0, tokens + n, &k);
    n += k;
    if (i < 2) {
      /* Meta's basic chat formatter strips surrounding message whitespace. */
      const char *begin = contents[i], *end = begin + strlen(begin);
      size_t skip;
      while ((skip = trim_space(begin, (size_t)(end - begin))))
        begin += skip;
      while (end > begin) {
        const char *last = end - 1;
        while (last > begin && ((unsigned char)*last & 0xc0) == 0x80)
          last--;
        if (!trim_space(last, (size_t)(end - last)))
          break;
        end = last;
      }
      size_t prefix_len = i == 0 ? strlen(prefix) : 0;
      char *content = memory_alloc(checked_add(prefix_len, (size_t)(end - begin)) + 1, 1, MEM_PROMPT);
      memcpy(content, prefix, prefix_len);
      memcpy(content + prefix_len, begin, (size_t)(end - begin));
      encode(t, content, 0, 0, tokens + n, &k);
      n += k;
      memory_free(content);
      tokens[n++] = t->eot_id;
    }
  }
  *count = n;
  return tokens;
}

void encode(Tokenizer *t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // Special IDs follow the original LLaMA 3 vocabulary.
  if (text == NULL) {
    fail("cannot encode NULL text");
  }

  if (strlen(text) > INT_MAX - 2)
    fail("prompt is too long");
  if (t->version) {
#ifdef LLAMA_PCRE2
    encode_ranked(t, text, bos, eos, tokens, n_tokens);
    return;
#else
    fail("versioned tokenizer requires PCRE2=1");
#endif
  }
  if (t->sorted_vocab == NULL) {
    // lazily malloc and sort the vocabulary
    t->sorted_vocab = memory_alloc(t->vocab_size, sizeof(TokenIndex), MEM_TOKENIZER);
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // Candidates can concatenate three maximum-length pieces. Reserve UTF-8
  // scratch space too, even when all vocabulary entries are single bytes.
  char *str_buffer = memory_alloc((size_t)t->max_token_length * 3 + 5, 1, MEM_PROMPT);
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=128000) token, if desired
  if ((bos || eos) && t->vocab_size <= EOS_ID)
    fail("tokenizer lacks LLaMA 3 special tokens");
  if (bos)
    tokens[(*n_tokens)++] = BOS_ID;

  // process the raw (UTF-8) byte sequence of the input string
  for (const char *c = text; *c != '\0'; c++) {

    // Start a new codepoint at ASCII or a UTF-8 leading byte.
    if ((*c & 0xC0) != 0x80) {
      str_len = 0;
    }

    // append the current byte to the buffer
    str_buffer[str_len++] = *c;
    str_buffer[str_len] = '\0';

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning str_buffer size.
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // ok c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for (size_t i = 0; i < str_len; i++) {
        /* Preserve the legacy byte fallback convention; not tiktoken parity. */
        int byte_id = (unsigned char)str_buffer[i] + 3;
        if (byte_id >= t->vocab_size)
          fail("tokenizer lacks byte fallback %d", byte_id);
        tokens[(*n_tokens)++] = byte_id;
      }
    }
    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair or triple each iteration, according to the scores in vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;
    int best_len = 2; // length of the best merge sequence (2 for pair, 3 for triple)

    // first, try to find the best pair to merge
    for (int i = 0; i < (*n_tokens - 1); i++) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    // if no pair was found, try to find the best triple to merge
    if (best_idx == -1) {
      for (int i = 0; i < (*n_tokens - 2); i++) {
        // check if we can merge the triple (tokens[i], tokens[i+1], tokens[i+2])
        sprintf(str_buffer, "%s%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]], t->vocab[tokens[i + 2]]);
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1 && t->vocab_scores[id] > best_score) {
          // this merge triple exists in vocab! record its score and position
          best_score = t->vocab_scores[id];
          best_id = id;
          best_idx = i;
          best_len = 3;
        }
      }
    }

    if (best_idx == -1) {
      break; // we couldn't find any more pairs or triples to merge, so we're done
    }

    // merge the consecutive pair or triple (best_idx, best_idx+1[, best_idx+2]) into new token best_id
    tokens[best_idx] = best_id;
    // delete token(s) at position best_idx+1 (and optionally best_idx+2), shift the entire sequence back
    for (int i = best_idx + 1; i < (*n_tokens - best_len + 1); i++) {
      tokens[i] = tokens[i + best_len - 1];
    }
    (*n_tokens) -= (best_len - 1); // token length decreased by the number of merged tokens minus one
  }

  // add optional EOS (=128001) token, if desired
  if (eos)
    tokens[(*n_tokens)++] = EOS_ID;

  memory_free(str_buffer);
}
