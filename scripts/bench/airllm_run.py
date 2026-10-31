import sys, time, resource, os
mode = sys.argv[1]
P = "Once upon a time, in a small village by the sea, there lived an old fisherman who"
N = 40
path = os.path.abspath("tinyllama")
if mode == "airllm":
    from airllm import AutoModel
    t0 = time.time()
    m = AutoModel.from_pretrained(path)
    t_load = time.time() - t0
    t0 = time.time()
    import mlx.core as mx
    toks = m.tokenizer([P], return_tensors="np", return_attention_mask=False, truncation=True, max_length=128, padding=False)
    t0 = time.time()
    out = m.generate(mx.array(toks["input_ids"]), max_new_tokens=N, use_cache=True, return_dict_in_generate=True)
    dt = time.time() - t0
    print("AIRLLM load %.1fs generate %.1fs for %d tokens = %.2f s/token; output: %r" % (t_load, dt, N, dt / N, str(out)[:160]))
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    t_load = time.time() - t0
    ids = tok(P, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        o = m.generate(**ids, max_new_tokens=N, do_sample=False)
    dt = time.time() - t0
    print("TORCH load %.1fs generate %.1fs for %d tokens = %.2f s/token; output: %r" % (t_load, dt, N, dt / N, tok.decode(o[0])[:120]))
print("peak RSS %.0f MiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576))
