#!/usr/bin/env python3
"""
Asks a Qwen3-VL model about a picture through the engine's chat protocol (greedy, so the answer can be compared with llama.cpp's llama-mtmd-cli on the same files):

    python3 test/vision_chat.py build/clutor-engine model.gguf mmproj.gguf picture.ppm ["question"] [max tokens]
"""
import base64, json, subprocess, sys, time

engine, model, mmproj, ppm = sys.argv[1:5]
question = sys.argv[5] if len(sys.argv) > 5 else "Describe this image briefly."
maxn = int(sys.argv[6]) if len(sys.argv) > 6 else 80
with open(ppm, "rb") as f:
    assert f.readline().strip() == b"P6"; w, h = map(int, f.readline().split()); f.readline(); rgb = f.read()
p = subprocess.Popen([engine, "chat", model, "--mmproj", mmproj, "--ctx", "2048"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def ev():
    return json.loads(p.stdout.readline())
ready = ev(); print("ready:", {k: ready[k] for k in ("arch", "nCtx", "vision")})
prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n"
t0 = time.time()
p.stdin.write(json.dumps({"op": "append", "text": prompt, "images": [{"w": w, "h": h, "rgb": base64.b64encode(rgb).decode()}]}) + "\n"); p.stdin.flush()
while True:
    e = ev()
    if e["ev"] == "vision": print(f"picture: {e['tokens']} tokens, {e['width']}x{e['height']}, encoded in {e['seconds']} s")
    elif e["ev"] == "error": print("ERROR", e["msg"]); sys.exit(1)
    elif e["ev"] == "appended": print(f"prompt: {e['tokens']} tokens in {e['seconds']} s (pos {e['pos']})"); break
p.stdin.write(json.dumps({"op": "generate", "max": maxn, "temp": 0}) + "\n"); p.stdin.flush()
out = ""
while True:
    e = ev()
    if e["ev"] == "t": out += e["s"]
    elif e["ev"] == "done": print(f"answer ({e['tokens']} tokens, {e['tokensPerSecond']:.2f} tok/s):\n{out.strip()}"); break
    elif e["ev"] == "error": print("ERROR", e["msg"]); sys.exit(1)
p.stdin.write('{"op":"quit"}\n'); p.stdin.flush(); p.wait(timeout=30)
