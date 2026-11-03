/* The engine's conversation API (see chat.c). One JSON object per line in (cl_chat_run), one JSON object per event out (the emit callback). */
#ifndef CL_CHAT_H
#define CL_CHAT_H
#include "forward.h"
#include "meta.h"
#include "tokenizer.h"
#include <stddef.h>
#include <stdint.h>

typedef struct cl_chat cl_chat;
typedef void (*cl_chat_emit)(void *user, const char *json_line);   /* one JSON object, no trailing newline; called on the thread running the operation */
typedef int (*cl_chat_poll)(void *user);                            /* optional: return 1 to stop the running operation (checked between tokens) */

/* argv[0] is the model path; the rest are the flags `clutor-engine chat` takes (--ctx N, --lock, --cache-mb N, ...). Emits the "ready" event. NULL (and err) on failure. */
cl_chat *cl_chat_open(int argc, const char *const *argv, cl_chat_emit emit, void *user, char *err, size_t errlen);
void cl_chat_set_poll(cl_chat *c, cl_chat_poll poll, void *user);
/* Runs one operation ({"op":"append"|"generate"|"reset"|"cancel"|"quit", ...}) to completion. Returns 1 for "quit", else 0. */
int cl_chat_run(cl_chat *c, const char *op_json);
/* Stops the running append or generate at the next token. Safe from any thread. */
void cl_chat_cancel(cl_chat *c);
void cl_chat_close(cl_chat *c);

/* Shared with the command line. Flags -> streaming / pinning options; on a bad flag, err is set and .policy is -1. */
cl_fwd_opts cl_fwd_opts_from_args(int argc, const char *const *argv, char *err, size_t errlen);
int cl_load_tokenizer(const char *path, cl_meta *meta, uint8_t **map, size_t *msize, cl_tok **tok, char *err, size_t errlen);
/* The machine, as the engine's portability layer sees it. */
uint64_t cl_system_ram(void);
int cl_system_cores(void);
#endif
