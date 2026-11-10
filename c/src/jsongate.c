#include "jsongate.h"
#include <string.h>

enum { M_VALUE, M_VALUE_OR_END, M_KEY_OR_END, M_KEY, M_COLON, M_AFTER, M_STR, M_ESC, M_UNI, M_NUM, M_LIT, M_DONE };
enum { N_MINUS, N_ZERO, N_INT, N_DOT, N_FRAC, N_E, N_ESIGN, N_EXP };

void cl_json_init(cl_json *j, int object_only) { memset(j, 0, sizeof *j); j->mode = M_VALUE; j->object_only = (uint8_t)(object_only != 0); }

static void after_value(cl_json *j) { j->mode = j->depth == 0 ? M_DONE : M_AFTER; }
static int is_ws(unsigned char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
static int is_hex(unsigned char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
static int push(cl_json *j, char k) { if (j->depth >= CL_JSON_DEPTH) return 0; j->stack[j->depth++] = (uint8_t)k; return 1; }

/* The start of a value (the character has already been checked not to be whitespace). */
static int start_value(cl_json *j, unsigned char c) {
  if (j->object_only && j->depth == 0 && c != '{') return 0;
  switch (c) {
    case '{': if (!push(j, '{')) return 0; j->mode = M_KEY_OR_END; return 1;
    case '[': if (!push(j, '[')) return 0; j->mode = M_VALUE_OR_END; return 1;
    case '"': j->mode = M_STR; j->is_key = 0; return 1;
    case 't': j->mode = M_LIT; j->left = 3; j->num = 0; return 1;   /* num holds which literal: 0 true, 1 false, 2 null */
    case 'f': j->mode = M_LIT; j->left = 4; j->num = 1; return 1;
    case 'n': j->mode = M_LIT; j->left = 3; j->num = 2; return 1;
    case '-': j->mode = M_NUM; j->num = N_MINUS; return 1;
    case '0': j->mode = M_NUM; j->num = N_ZERO; return 1;
    default: if (c >= '1' && c <= '9') { j->mode = M_NUM; j->num = N_INT; return 1; } return 0;
  }
}

static int step(cl_json *j, unsigned char c);

/* A number ends at the first character that cannot continue it; that character is then read as what follows a value. */
static int num_end_ok(const cl_json *j) { return j->num == N_ZERO || j->num == N_INT || j->num == N_FRAC || j->num == N_EXP; }

static int step(cl_json *j, unsigned char c) {
  switch (j->mode) {
    case M_DONE: return is_ws(c);
    case M_VALUE: if (is_ws(c)) return 1; return start_value(j, c);
    case M_VALUE_OR_END: if (is_ws(c)) return 1; if (c == ']') { j->depth--; after_value(j); return 1; } return start_value(j, c);
    case M_KEY_OR_END: if (is_ws(c)) return 1; if (c == '}') { j->depth--; after_value(j); return 1; } /* fall through */
    case M_KEY: if (is_ws(c)) return 1; if (c == '"') { j->mode = M_STR; j->is_key = 1; return 1; } return 0;
    case M_COLON: if (is_ws(c)) return 1; if (c == ':') { j->mode = M_VALUE; return 1; } return 0;
    case M_AFTER: {
      if (is_ws(c)) return 1;
      const uint8_t top = j->stack[j->depth - 1];
      if (c == ',') { j->mode = top == '{' ? M_KEY : M_VALUE; return 1; }
      if ((c == '}' && top == '{') || (c == ']' && top == '[')) { j->depth--; after_value(j); return 1; }
      return 0;
    }
    case M_STR:
      if (c == '"') { if (j->is_key) j->mode = M_COLON; else after_value(j); return 1; }
      if (c == '\\') { j->mode = M_ESC; return 1; }
      return c >= 0x20;   /* any byte of UTF-8 text is fine; raw control characters are not */
    case M_ESC:
      if (c == 'u') { j->mode = M_UNI; j->left = 4; return 1; }
      if (strchr("\"\\/bfnrt", c) && c != 0) { j->mode = M_STR; return 1; }
      return 0;
    case M_UNI: if (!is_hex(c)) return 0; if (--j->left == 0) j->mode = M_STR; return 1;
    case M_LIT: {
      static const char *const lit[3] = {"true", "false", "null"};
      const char *w = lit[j->num];
      size_t at = strlen(w) - j->left;
      if (c != (unsigned char)w[at]) return 0;
      if (--j->left == 0) after_value(j);
      return 1;
    }
    case M_NUM:
      switch (j->num) {
        case N_MINUS: if (c == '0') { j->num = N_ZERO; return 1; } if (c >= '1' && c <= '9') { j->num = N_INT; return 1; } return 0;
        case N_ZERO: case N_INT:
          if (j->num == N_INT && c >= '0' && c <= '9') return 1;
          if (c == '.') { j->num = N_DOT; return 1; }
          if (c == 'e' || c == 'E') { j->num = N_E; return 1; }
          break;
        case N_DOT: if (c >= '0' && c <= '9') { j->num = N_FRAC; return 1; } return 0;
        case N_FRAC:
          if (c >= '0' && c <= '9') return 1;
          if (c == 'e' || c == 'E') { j->num = N_E; return 1; }
          break;
        case N_E: if (c == '+' || c == '-') { j->num = N_ESIGN; return 1; } if (c >= '0' && c <= '9') { j->num = N_EXP; return 1; } return 0;
        case N_ESIGN: if (c >= '0' && c <= '9') { j->num = N_EXP; return 1; } return 0;
        case N_EXP: if (c >= '0' && c <= '9') return 1; break;
        default: return 0;
      }
      if (!num_end_ok(j)) return 0;
      after_value(j);
      return step(j, c);
    default: return 0;
  }
}

int cl_json_feed(cl_json *j, const char *s, size_t n) {
  if (j->bad) return 0;
  for (size_t i = 0; i < n; i++) if (!step(j, (unsigned char)s[i])) { j->bad = 1; return 0; }
  return 1;
}
int cl_json_done(const cl_json *j) { return !j->bad && (j->mode == M_DONE || (j->mode == M_NUM && j->depth == 0 && num_end_ok(j))); }
