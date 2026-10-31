/*
 * Byte-level BPE tokenizer (GPT-2 family) built from a GGUF file's own tokenizer tables. The pre-tokenizers implemented are the ones llama.cpp
 * calls "olmo" and "qwen2". Letter, number and punctuation classes are exact for ASCII and approximate above U+007F (see tokenizer.c), so non-English
 * text can tokenize differently from llama.cpp; `make verify-llama` checks agreement on a fixed set of prompts.
 */
#ifndef CL_TOKENIZER_H
#define CL_TOKENIZER_H
#include "meta.h"
#include <stddef.h>
#include <stdint.h>

typedef struct cl_tok cl_tok;

cl_tok *cl_tok_load(const cl_meta *m, char *err, size_t errlen);
void cl_tok_free(cl_tok *t);
int cl_tok_vocab_size(const cl_tok *t);
int cl_tok_bos(const cl_tok *t);
int cl_tok_eos(const cl_tok *t);
int cl_tok_add_bos(const cl_tok *t);
/* Encodes text. With parse_special, control tokens such as <|endoftext|> written in the text become their single token. Returns the number of
 * tokens written, or -1 when `cap` is too small. */
int cl_tok_encode(const cl_tok *t, const char *text, size_t len, int parse_special, int32_t *out, int cap);
/* The bytes of one token (control tokens give their text). Returns the length written, at most cap. */
int cl_tok_piece(const cl_tok *t, int id, char *out, int cap);
#endif
