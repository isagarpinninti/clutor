#ifndef CL_JSONGATE_H
#define CL_JSONGATE_H
#include <stddef.h>
#include <stdint.h>

/* A streaming JSON checker for constrained decoding: it knows whether the text fed so far is the beginning of a JSON value, so the sampler can refuse
 * any token that would make the reply unparseable. The state is a small plain struct: copy it to try a token without committing to it. */
#define CL_JSON_DEPTH 64
typedef struct { uint8_t mode, num, left, is_key, object_only, depth, bad; uint8_t stack[CL_JSON_DEPTH]; } cl_json;

/* object_only: the top-level value must be an object (the agent's tool calls are). */
void cl_json_init(cl_json *j, int object_only);
/* Feeds bytes. Returns 1 while the text is still a valid beginning, 0 as soon as it is not (the state is then unusable). */
int cl_json_feed(cl_json *j, const char *s, size_t n);
/* 1 once the top-level value is complete (only whitespace may follow). */
int cl_json_done(const cl_json *j);
#endif
