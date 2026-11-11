# Directions

Where Clutor could go next, written so each idea can be checked. Nothing here is built unless it says so. The order is the order I would do them.

## 1. One kernel seam for every kind of hardware (partly built)

Every matrix product in the engine goes through two functions, `cl_matvec` and `cl_matvec_batch` (`c/src/quant.h`). That is the seam. A **backend** is a table with a name, a question ("do you handle this weight type?") and the batch function; the CPU kernels are the default backend, and another backend is chosen with `CLUTOR_BACKEND=<name>`. Built: the table, the registry and a test that routes products through a counting backend and checks the numbers match (`c/src/backend.h`, `test_backend` in `c/test/test_engine.c`). Not built: any backend other than the CPU.

What a GPU backend has to get right, from reading the code:

- **Weights.** The engine memory-maps the model file. On Apple silicon a Metal buffer can wrap the same pages without a copy (`newBufferWithBytesNoCopy`, which needs page-aligned address and length), so a model that streams experts from disk keeps streaming; a discrete GPU needs the weights copied into its own memory, which changes the out-of-core story and needs its own design.
- **Exactness.** The CPU paths are bit-identical to each other by test. A GPU will not be (floating-point order differs), so the test becomes "same top token and logits within a stated tolerance", as the engine already uses against llama.cpp.
- **Where the gain is.** Measured on the M2 (`docs/BENCHMARKS.md`): llama.cpp on the GPU reads a 0.8 GB model about 1.7 times faster than its own CPU path while decoding, and about 6 times faster while processing a prompt. Decoding is limited by memory bandwidth, which the CPU and GPU share, so the decode gain has a ceiling; prompt processing is compute-bound, so that is where a Metal backend would pay most.
- **The work.** A Metal backend is a matter of weeks, not days: one compute kernel per weight type (K-quants and IQ formats are the awkward ones), command-buffer batching so a layer is one submission, and a CPU fallback for types without a kernel. The CPU+GPU split (the router and sampling stay on the CPU) is already how the code is organised.

NPUs and other accelerators would sit behind the same seam. Whether they help depends on whether their runtime lets you hand it quantised weights in the layouts GGUF uses; that has to be checked per device, not assumed.

## 2. A mesh of machines (design only; I would not start it yet)

The idea: several computers each hold some layers, or some experts, and pass the activations along, so a model too big for any one fits in all of them together.

What the numbers say (computed, not measured):

- Splitting by layer sends one activation vector between machines per token: for a model with a hidden size of 4096 that is 16 KB at 32-bit floats. A gigabit network moves that in well under a millisecond, so **bandwidth is not the problem**; **latency is**. Every hop adds a round trip (about 0.2 to 1 ms on a good wired network, several milliseconds over Wi-Fi), and a token crosses every machine once, so four machines on Wi-Fi can add 10 ms or more per token before any computing.
- Splitting by expert (each machine owns some experts) needs two network trips per layer, because the router's choice must go out and the results come back. With 48 layers that is about 100 trips per token, which is far worse.
- So the realistic form is **layer split across two to four wired machines for a model that fits nowhere alone**, giving each token a few extra milliseconds. It would not speed up a model that already fits on one machine.
- Prior art exists (for example `exo` and llama.cpp's RPC mode), so this is a feature to build only if the single-machine engine is already strong.

A first experiment that would settle whether it is worth doing: run the engine's layers 0..n/2 on one machine and the rest on another over a wire, with a toy protocol, on the 1B model, and measure tokens per second against one machine. That is about a week of work. It is not started.

## 3. Editions

Free (MIT): the engine, the planner, the CLI, the app and the agent, as now. A paid edition only makes sense if it sells something the free one does not have, and today that list is not long enough to justify one. Candidates, in the order I would believe in them: a signed, auto-updating installer for companies that cannot build from source; a managed policy file (section 4); support. A hosted service is not on the list because the product's point is that nothing leaves the machine.

Not decided and not built. I would wait until more than one platform is tested and a few outside users exist.

## 4. Company use: a policy file (partly built)

A company wants to say what the agent may do without trusting each laptop's settings. Built, in the command line and the local server's agent: `clutor.policy.json` (in the project folder, or the file named by `CLUTOR_POLICY`) with two switches, `requireSandbox` (commands are refused when the computer has no OS sandbox) and `forbidAutoApprove` (the configuration key and `--yes` are ignored, so the person is always asked). Unknown keys and wrong types are errors, so a typo cannot silently turn protection off. Tested in `packages/core/test/policy.test.ts` and `node.test.ts`.

Not built: the desktop app does not read the file yet; there is no switch to forbid cloud providers or to fix the permission level; nothing signs the file. And it cannot stop a person who can edit files on their own computer from deleting it or changing `CLUTOR_POLICY`; for that a company needs its device-management tooling to place a read-only file.

## 5. The name

I have not changed it. A name should come from what the product is once the direction is settled, and it is your decision, not mine. If you want candidates, I would look for a name that says "bigger than your memory" or "honest numbers" and check that it is free as an npm package, a GitHub organisation and a domain before falling for it. Say so and I will make a list with those checks done.
