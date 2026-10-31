#include "runtime.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int compare_tokens(const void *a, const void *b) {
  return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, const char *tokenizer_path, int vocab_size) {
  t->vocab_size = vocab_size;
  t->vocab = memory_alloc(vocab_size, sizeof(char *), MEM_TOKENIZER);
  t->vocab_scores = memory_alloc(vocab_size, sizeof(float), MEM_TOKENIZER);
  t->sorted_vocab = NULL;
  memset(t->byte_pieces, 0, sizeof(t->byte_pieces));
  for (int i = 0; i < 256; i++)
    t->byte_pieces[i * 2] = (unsigned char)i;
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file)
    fail("cannot open tokenizer '%s': %s", tokenizer_path, strerror(errno));
  if (fread(&t->max_token_length, 4, 1, file) != 1 || !t->max_token_length || t->max_token_length > 1048576)
    fail("invalid tokenizer header in '%s' (maximum token length must be 1..1048576)", tokenizer_path);
  for (int i = 0; i < vocab_size; i++) {
    int len;
    if (fread(t->vocab_scores + i, 4, 1, file) != 1 || fread(&len, 4, 1, file) != 1 || len <= 0 ||
        (unsigned)len > t->max_token_length || !isfinite(t->vocab_scores[i]))
      fail("invalid/truncated tokenizer '%s' at token %d", tokenizer_path, i);
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
}

const char *decode(Tokenizer *t, int token) {
  if (token < 0 || token >= t->vocab_size)
    fail("invalid decoded token %d", token);
  char *piece = t->vocab[token];

  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
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

void encode(Tokenizer *t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // Special IDs follow the original LLaMA 3 vocabulary.
  if (text == NULL) {
    fail("cannot encode NULL text");
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
