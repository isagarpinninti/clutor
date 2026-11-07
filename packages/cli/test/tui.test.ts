import { existsSync, mkdtempSync, readFileSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { afterEach, describe, expect, it } from "vitest";
import { Region, fit, makeStyle, stripAnsi, width, wrap } from "../src/tui/ansi";
import { Editor, layout, type Cmd } from "../src/tui/editor";
import { checkoutRoot, discoverModels, findGguf, modelFolders } from "../src/tui/discover";
import { KeyParser } from "../src/tui/keys";
import { MarkdownStream } from "../src/tui/markdown";
import { spinnerLine } from "../src/tui/spinner";
import { streamChat } from "../src/tui/stream";
import { expandFiles, runTui } from "../src/tui/app";
import { callTitle, renderPatch, resultSummary, withoutThinking } from "../src/tui/agent";
import { highlightLine } from "../src/tui/highlight";
import { THEMES } from "../src/tui/ansi";

const plain = makeStyle(false);
const color = makeStyle(true);

describe("ansi", () => {
  it("measures display width: wide characters count twice, escapes not at all", () => {
    expect(width("abc")).toBe(3);
    expect(width("日本")).toBe(4);
    expect(width(color.bold("hi"))).toBe(2);
  });
  it("wraps at spaces with a hanging indent and never exceeds the width", () => {
    const rows = wrap("the quick brown fox jumps over the lazy dog", 16, 2);
    expect(rows.length).toBeGreaterThan(2);
    for (const r of rows) expect(width(r)).toBeLessThanOrEqual(16);
    expect(rows[1]?.startsWith("  ")).toBe(true);
    expect(stripAnsi(rows.join(" ").replace(/\s+/g, " ").trim())).toBe("the quick brown fox jumps over the lazy dog");
  });
  it("splits a word longer than a row", () => {
    const rows = wrap("a".repeat(40), 16);
    expect(rows.every((r) => width(r) <= 16)).toBe(true);
    expect(rows.join("")).toBe("a".repeat(40));
  });
  it("keeps styles open across a wrapped break", () => {
    const rows = wrap(color.bold("one two three four five six seven eight"), 12);
    expect(rows.length).toBeGreaterThan(1);
    expect(rows[0]).toContain("\x1b[1m");
  });
  it("cuts a line to a width and closes its style", () => {
    const f = fit(color.accent("abcdefghij"), 4);
    expect(stripAnsi(f)).toBe("abcd");
    expect(f.endsWith("\x1b[0m")).toBe(true);
  });
  it("has no escapes when colour is off", () => { expect(plain.bold("x")).toBe("x"); expect(plain.accent("x")).toBe("x"); });
});

describe("Region", () => {
  it("redraws in place: moves up to its top, clears, and rewrites", () => {
    let buf = "";
    const r = new Region({ write: (s: string) => { buf += s; return true; }, columns: 40 });
    r.draw(["a", "b", "c"], { row: 1, col: 3 });
    expect(buf).toContain("a\nb\nc");
    buf = "";
    r.draw(["x"]);
    expect(buf).toContain("\x1b[1A");
    expect(buf).toContain("\x1b[J");
  });
});

describe("KeyParser", () => {
  const keys = (s: string) => new KeyParser().feed(s).map((k) => (k.text ? `${k.name}:${k.text}` : k.name + (k.alt ? "+alt" : "") + (k.ctrl ? "+ctrl" : "")));
  it("reads typed text, enter, backspace and arrows", () => {
    expect(keys("hi")).toEqual(["char:hi"]);
    expect(keys("\r")).toEqual(["enter"]);
    expect(keys("\x7f")).toEqual(["backspace"]);
    expect(keys("\x1b[A\x1b[B\x1b[C\x1b[D")).toEqual(["up", "down", "right", "left"]);
    expect(keys("\x1b[3~")).toEqual(["delete"]);
  });
  it("reads control keys and a lone escape", () => {
    expect(keys("\x03")).toEqual(["ctrl-c+ctrl"]);
    expect(keys("\x01")).toEqual(["ctrl-a+ctrl"]);
    expect(keys("\x1b")).toEqual(["escape"]);
  });
  it("treats alt-enter and kitty shift-enter as a new line, and word movement", () => {
    expect(keys("\x1b\r")).toEqual(["newline"]);
    expect(keys("\x1b[13;2u")).toEqual(["newline"]);
    expect(keys("\x1b[1;5C")).toEqual(["right+ctrl"]);
  });
  it("gathers a bracketed paste, even across chunks, keeping its newlines", () => {
    const p = new KeyParser();
    expect(p.feed("\x1b[200~line1\nli")).toEqual([]);
    expect(p.feed("ne2\x1b[201~")).toEqual([{ name: "paste", text: "line1\nline2" }]);
  });
});

describe("Editor", () => {
  const cmds: Cmd[] = [{ name: "help", help: "h" }, { name: "model", help: "m" }, { name: "think", args: "[on|off]", help: "t" }, { name: "temp", args: "<n>", help: "x" }];
  const type = (e: Editor, s: string) => e.handle({ name: "char", text: s });
  it("edits text and submits it, remembering history", () => {
    const e = new Editor([], cmds);
    type(e, "hello"); e.handle({ name: "left" }); e.handle({ name: "backspace" });
    expect(e.text).toBe("helo");
    expect(e.handle({ name: "enter" })).toEqual({ type: "submit", text: "helo" });
    expect(e.text).toBe("");
    e.handle({ name: "up" });
    expect(e.text).toBe("helo");
  });
  it("walks history and returns to the draft", () => {
    const e = new Editor(["one", "two"], cmds);
    type(e, "dr"); e.handle({ name: "up" }); expect(e.text).toBe("two");
    e.handle({ name: "up" }); expect(e.text).toBe("one");
    e.handle({ name: "down" }); e.handle({ name: "down" }); expect(e.text).toBe("dr");
  });
  it("makes new lines with backslash-enter and keeps them in one message", () => {
    const e = new Editor([], cmds);
    type(e, "a\\"); e.handle({ name: "enter" }); type(e, "b");
    expect(e.text).toBe("a\nb");
    expect(e.handle({ name: "enter" })).toEqual({ type: "submit", text: "a\nb" });
  });
  it("filters the slash menu and completes or runs the selection", () => {
    const e = new Editor([], cmds);
    type(e, "/t");
    expect(e.matches().map((c) => c.name)).toEqual(["think", "temp"]);
    e.handle({ name: "down" });
    e.handle({ name: "tab" });
    expect(e.text).toBe("/temp ");
    const f = new Editor([], cmds);
    type(f, "/he");
    expect(f.handle({ name: "enter" })).toEqual({ type: "submit", text: "/help" });
  });
  it("asks twice before ctrl-c exits, and ctrl-d exits only when empty", () => {
    const e = new Editor([], cmds);
    expect(e.handle({ name: "ctrl-c" })?.type).toBe("notice");
    expect(e.handle({ name: "ctrl-c" })).toEqual({ type: "exit" });
    const f = new Editor([], cmds); type(f, "x");
    expect(f.handle({ name: "ctrl-d" })).toBeUndefined();
    expect(f.handle({ name: "ctrl-c" })).toBeUndefined();
    expect(f.text).toBe("");
    expect(f.handle({ name: "ctrl-d" })).toEqual({ type: "exit" });
  });
  it("draws a box, puts the cursor after the text, and fits every row", () => {
    const e = new Editor([], cmds);
    type(e, "hello world");
    const v = e.view(plain, 30, { left: "model", right: "ctx" });
    expect(v.lines[0]).toBe(`╭${"─".repeat(28)}╮`);
    expect(v.lines[1]).toBe(`│ ❯ hello world${" ".repeat(13)}│`);
    expect(v.lines[2]).toBe(`╰${"─".repeat(28)}╯`);
    expect(v.cursor).toEqual({ row: 1, col: 15 });
    for (const l of v.lines) expect(width(l)).toBeLessThanOrEqual(30);
  });
  it("wraps long input over several rows and tracks the cursor", () => {
    const e = new Editor([], cmds);
    type(e, "x".repeat(50));
    const v = e.view(plain, 24, { left: "", right: "" });
    expect(layout(e.text, 17).length).toBe(3);
    expect(v.cursor.row).toBe(3);
    for (const l of v.lines) expect(width(l)).toBeLessThanOrEqual(24);
  });
  it("shows the menu under the box", () => {
    const e = new Editor([], cmds);
    type(e, "/");
    const v = e.view(plain, 60, { left: "", right: "" });
    expect(v.lines.slice(3).join("\n")).toContain("/help");
    expect(v.lines.slice(3).join("\n")).toContain("/think [on|off]");
  });
  it("scrolls a long menu and says how many more there are", () => {
    const many: Cmd[] = Array.from({ length: 20 }, (_, i) => ({ name: `c${i}`, help: "x" }));
    const e = new Editor([], many);
    type(e, "/");
    expect(e.view(plain, 60, { left: "", right: "" }).lines.join("\n")).toMatch(/↓ 12 more/);
  });
});

describe("MarkdownStream", () => {
  const render = (src: string, chunk = 3): string[] => {
    const md = new MarkdownStream(plain, () => 60);
    const rows: string[] = [];
    for (let i = 0; i < src.length; i += chunk) rows.push(...md.push(src.slice(i, i + chunk)).done);
    rows.push(...md.end().done);
    return rows;
  };
  it("renders lists, headings and quotes the same whatever the chunking", () => {
    const src = "# Title\n\n- one\n- two\n1. first\n> quoted\n";
    expect(render(src, 1)).toEqual(render(src, 100));
    expect(render(src)).toContain("• one");
    expect(render(src)).toContain("1. first");
    expect(render(src)).toContain("▎ quoted");
  });
  it("draws code fences with a gutter and does not interpret Markdown inside them", () => {
    const rows = render("```ts\nconst a = **x**;\n```\n");
    expect(rows[0]).toBe("╭─ ts ");
    expect(rows[1]).toBe("│ const a = **x**;");
    expect(rows[2]).toBe("╰─");
  });
  it("styles inline code, bold and italic, and leaves unclosed markers alone while streaming", () => {
    const md = new MarkdownStream(color, () => 60);
    expect(stripAnsi(md.inline("a `b` **c** *d*"))).toBe("a b c d");
    expect(md.inline("**c** *d*")).toContain("\x1b[1m");
    expect(md.inline("half **open")).toBe("half **open");
  });
  it("shows the unfinished line as live rows", () => {
    const md = new MarkdownStream(plain, () => 60);
    const r = md.push("hello wor");
    expect(r.done).toEqual([]);
    expect(r.live).toEqual(["hello wor"]);
  });
  it("lays a table out in aligned columns", () => {
    const rows = render("| a | bb |\n|---|----|\n| ccc | d |\n");
    expect(rows.length).toBe(3);
    expect(rows[0]).toContain("a   │ bb");
    expect(rows[1]).toContain("┼");
  });
  it("wraps a long paragraph to the width", () => {
    const rows = render(`${"word ".repeat(40)}\n`);
    expect(rows.length).toBeGreaterThan(2);
    for (const r of rows) expect(width(r)).toBeLessThanOrEqual(60);
  });
});

describe("spinner", () => {
  it("animates: different frames at different times, with the numbers shown", () => {
    const a = spinnerLine(plain, 0, "Pondering", ["3s", "↓ 10 tokens"]);
    const b = spinnerLine(plain, 400, "Pondering", ["3s"]);
    expect(a).toContain("Pondering…");
    expect(a).toContain("(3s · ↓ 10 tokens)");
    expect(a[0]).not.toBe(b[0]);
  });
});

function fakeSse(chunks: string[]): typeof fetch {
  return (async () => new Response(new ReadableStream({ start(c) { for (const s of chunks) c.enqueue(new TextEncoder().encode(s)); c.close(); } }), { status: 200, headers: { "content-type": "text/event-stream" } })) as typeof fetch;
}
const delta = (d: object, extra: object = {}) => `data: ${JSON.stringify({ choices: [{ delta: d }], ...extra })}\n\n`;

describe("streamChat", () => {
  it("yields content and reasoning in order with progress, usage and timings", async () => {
    const f = fakeSse([
      `data: ${JSON.stringify({ prompt_progress: { total: 10, processed: 5 }, choices: [{ delta: {} }] })}\n\n`,
      delta({ reasoning_content: "hm" }), delta({ content: "Hel" }), delta({ content: "lo" }),
      `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 10, completion_tokens: 2 }, timings: { predicted_per_second: 12.5, prompt_per_second: 80 } })}\n\n`, "data: [DONE]\n\n",
    ]);
    const ev = [];
    for await (const e of streamChat({ baseUrl: "http://x/v1", model: "m", messages: [], fetchFn: f })) ev.push(e);
    expect(ev.map((e) => e.type)).toEqual(["progress", "reasoning", "content", "content", "done"]);
    expect(ev[4]).toMatchObject({ finish: "stop", promptTokens: 10, completionTokens: 2, tokensPerSecond: 12.5, promptPerSecond: 80 });
  });
  it("turns an HTTP error into a readable message", async () => {
    const f = (async () => new Response(JSON.stringify({ error: { message: "context too long" } }), { status: 400 })) as typeof fetch;
    await expect((async () => { for await (const _ of streamChat({ baseUrl: "http://x/v1", model: "m", messages: [], fetchFn: f })) void _; })()).rejects.toThrow(/400: context too long/);
  });
  it("asks for no thinking when told to", async () => {
    let body = "";
    const f = (async (_u: unknown, init: { body: string }) => { body = init.body; return new Response("data: [DONE]\n\n", { status: 200 }); }) as unknown as typeof fetch;
    for await (const _ of streamChat({ baseUrl: "http://x/v1", model: "m", messages: [], thinking: false, fetchFn: f })) void _;
    expect(JSON.parse(body).chat_template_kwargs).toEqual({ enable_thinking: false });
  });
});

describe("files", () => {
  const dirs: string[] = [];
  const tmp = () => { const d = mkdtempSync(join(tmpdir(), "clt-tui-")); dirs.push(d); return d; };
  afterEach(() => { for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true }); });
  it("finds models in nested folders and skips projectors and later split parts", () => {
    const d = tmp();
    mkdirSync(join(d, "Qwen", "x"), { recursive: true });
    for (const f of ["Qwen/x/a-Q4.gguf", "b.gguf", "mmproj-b.gguf", "c-00002-of-00003.gguf", "c-00001-of-00003.gguf", "notes.txt"]) writeFileSync(join(d, f), "x");
    expect(findGguf(d).map((m) => m.name).sort()).toEqual(["a-Q4", "b", "c"]);
  });
  it("looks beyond the current folder: the checkout's models folders and folders remembered from earlier choices", () => {
    const here = tmp(), elsewhere = tmp();
    writeFileSync(join(elsewhere, "remembered.gguf"), "x");
    expect(discoverModels(here, "models")).toEqual(discoverModels(here, "models").filter(() => true));   // nothing under `here`, whatever else is found
    expect(discoverModels(here, "models", [elsewhere]).map((m) => m.name)).toContain("remembered");
    const folders = modelFolders(here, "models", [elsewhere]);
    expect(folders).toContain(elsewhere);
    expect(folders).toContain(join(here, "Models"));
    const co = checkoutRoot();
    if (co) { expect(folders).toContain(join(co, "models")); expect(folders).toContain(join(co, "Models")); }
  });
  it("attaches @files inside the folder only", () => {
    const d = tmp();
    writeFileSync(join(d, "a.txt"), "line1\nline2");
    const r = expandFiles("look at @a.txt and @../etc/passwd and @missing", d);
    expect(r.attached).toEqual(["a.txt (2 lines)"]);
    expect(r.text).toContain('<file path="a.txt">');
    expect(r.text).not.toContain("passwd>");
  });
});

describe("the interactive session, against a fake model server", () => {
  const cleanup: Array<() => void> = [];
  afterEach(() => { for (const c of cleanup.splice(0)) c(); });

  it("chats: streams a reply, shows it, keeps the history, runs slash commands and exits", async () => {
    const seen: Array<{ messages: Array<{ role: string; content: string }>; chat_template_kwargs?: unknown }> = [];
    const server: Server = createServer((req, res) => {
      let b = ""; req.on("data", (c) => (b += c));
      req.on("end", () => {
        seen.push(JSON.parse(b));
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.write(delta({ content: "Hello " }));
        res.write(delta({ content: "**there**" }));
        res.write(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 7, completion_tokens: 2 }, timings: { predicted_per_second: 9.5 } })}\n\n`);
        res.end("data: [DONE]\n\n");
      });
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    cleanup.push(() => server.close());
    const port = (server.address() as { port: number }).port;
    const d = mkdtempSync(join(tmpdir(), "clt-tui-e2e-"));
    cleanup.push(() => rmSync(d, { recursive: true, force: true }));
    writeFileSync(join(d, "clutor.config.json"), JSON.stringify({ backend: { kind: "openai", baseUrl: `http://127.0.0.1:${port}/v1`, model: "fake" } }));

    const stdin = new PassThrough() as unknown as NodeJS.ReadStream;
    const stdout = new PassThrough() as unknown as NodeJS.WriteStream;
    (stdout as unknown as { columns: number }).columns = 80;
    let screen = "";
    stdout.on("data", (c) => (screen += String(c)));
    const ran: string[][] = [];
    const done = runTui({
      io: { out: () => undefined, err: () => undefined, cwd: d }, version: "9.9.9", argv: [], stdin, stdout, color: false, home: d,
      run: async (argv) => { ran.push(argv); return 0; },
    });
    const send = async (s: string) => { stdin.emit("data", s); await new Promise((r) => setTimeout(r, 250)); };
    await new Promise((r) => setTimeout(r, 150));
    await send("hi there\r");
    await send("/think\r");
    await send("again\r");
    await send("/status\r");
    await send("/cost\r");
    await send("/exit\r");
    expect(await done).toBe(0);

    const text = stripAnsi(screen);
    expect(text).toContain("Clutor v9.9.9");
    expect(text).toContain("Welcome back!");
    expect(text).toContain("● Hello there");
    expect(text).toContain("2 tokens · 9.5 tok/s");
    expect(text).toContain("thinking on");
    expect(seen.length).toBe(2);
    expect(seen[0]?.messages.map((m) => m.role)).toEqual(["system", "user"]);
    expect(seen[1]?.messages.map((m) => m.role)).toEqual(["system", "user", "assistant", "user"]);
    expect(seen[1]?.messages[2]?.content).toBe("Hello **there**");
    expect(seen[1]?.chat_template_kwargs).toEqual({ enable_thinking: true });
    expect(ran).toEqual([["status"]]);
    expect(text).toContain("Goodbye. 4 tokens generated in 2 replies.");
  });
});

describe("shift+tab", () => {
  it("is reported as a request to switch mode, not as menu completion", () => {
    expect(new KeyParser().feed("\x1b[Z")).toEqual([{ name: "backtab" }]);
    const e = new Editor([], [{ name: "help", help: "h" }]);
    expect(e.handle({ name: "backtab" })).toEqual({ type: "cycle-mode" });
  });
});

describe("themes", () => {
  it("every theme renders, and mono uses no colour", () => {
    for (const t of THEMES) {
      const s = makeStyle(true, t);
      expect(stripAnsi(s.accent("x"))).toBe("x");
      if (t === "mono") expect(s.tok("str", "x")).toBe("x");
      else expect(s.tok("str", "x")).toContain("\x1b[38;5;");
    }
    expect(makeStyle(true, "dark").accent("x")).not.toBe(makeStyle(true, "light").accent("x"));
  });
  it("can be switched in place without rebuilding what holds the style", () => {
    const s = makeStyle(true, "dark");
    const before = s.accent("x");
    Object.assign(s, makeStyle(true, "ocean"));
    expect(s.accent("x")).not.toBe(before);
  });
});

describe("syntax highlighting", () => {
  const c = makeStyle(true);
  it("colours keywords, strings, numbers, calls and comments, and keeps the text intact", () => {
    const line = 'const total = add(1, "two"); // done';
    const h = highlightLine(line, "ts", c);
    expect(stripAnsi(h)).toBe(line);
    expect(h).toContain(c.tok("kw", "const"));
    expect(h).toContain(c.tok("fn", "add"));
    expect(h).toContain(c.tok("num", "1"));
    expect(h).toContain(c.tok("str", '"two"'));
    expect(h).toContain(c.tok("com", "// done"));
  });
  it("handles python comments and JSON keys, and leaves unknown languages plain", () => {
    expect(highlightLine("x = 1  # note", "python", c)).toContain(c.tok("com", "# note"));
    expect(highlightLine('  "name": "x"', "json", c)).toContain(c.tok("fn", '"name"'));
    expect(highlightLine("whatever 1", "klingon", c)).toBe(c.code("whatever 1"));
    expect(highlightLine("let x", "ts", plain)).toBe(plain.code("let x"));
  });
  it("does not treat a number inside a word as a number", () => {
    const h = highlightLine("abc123 = 5", "py", c);
    expect(h).not.toContain(c.tok("num", "123"));
    expect(h).toContain(c.tok("num", "5"));
  });
  it("is used inside code blocks, whatever the chunking", () => {
    const md = new MarkdownStream(c, () => 60);
    const rows = [...md.push("```ts\nconst a = 1;\n```\n").done];
    expect(rows[1]).toContain(c.tok("kw", "const"));
  });
});

describe("agent display", () => {
  it("titles tool calls plainly: Read(file), Update(file), Run(command)", () => {
    expect(callTitle("read_file", { path: "a.ts" })).toBe("Read(a.ts)");
    expect(callTitle("edit_file", { path: "a.ts" })).toBe("Update(a.ts)");
    expect(callTitle("run_command", { command: "npm test" })).toBe("Run(npm test)");
    expect(callTitle("search", { pattern: "foo", dir: "src" })).toBe('Search("foo" in src)');
    expect(callTitle("finish", {})).toBeUndefined();
  });
  it("summarises results in a few words", () => {
    expect(resultSummary("read_file", true, "    1| a\n    2| b\n    3| ")).toBe("Read 2 lines");
    expect(resultSummary("read_file", true, "    1| a\n    2| ")).toBe("Read 1 line");
    expect(resultSummary("list_files", true, "a\nb\nc")).toBe("3 files");
    expect(resultSummary("search", true, "a.ts:1: x\nb.ts:2: y")).toBe("2 matches");
    expect(resultSummary("read_file", false, "x.ts does not exist")).toBe("x.ts does not exist");
  });
  it("colours a diff with line numbers and cuts long ones", () => {
    const patch = "--- a/f\n+++ b/f\n@@ -1,3 +1,3 @@\n keep\n-old\n+new\n tail\n";
    const rows = renderPatch(plain, patch).map(stripAnsi);
    expect(rows).toEqual(["    1   keep", "    2 - old", "    2 + new", "    3   tail"]);
    const big = `@@ -1,100 +1,100 @@\n${Array.from({ length: 100 }, (_, i) => `+l${i}`).join("\n")}\n`;
    const cut = renderPatch(plain, big, 10);
    expect(cut.length).toBe(11);
    expect(stripAnsi(cut[10] as string)).toContain("90 more lines");
  });
  it("turns thinking off in every request body it forwards", async () => {
    let body = "";
    const f = withoutThinking((async (_u: unknown, init: { body: string }) => { body = init.body; return new Response("{}"); }) as unknown as typeof fetch);
    await f("http://x", { method: "POST", body: JSON.stringify({ model: "m" }) });
    expect(JSON.parse(body)).toEqual({ model: "m", chat_template_kwargs: { enable_thinking: false } });
  });
});

describe("the coding agent inside the session", () => {
  const cleanup: Array<() => void> = [];
  afterEach(() => { for (const c of cleanup.splice(0)) c(); });

  it("shows each tool call, shows the diff, and writes the file only after you say yes", async () => {
    const script = [
      { tool: "read_file", args: { path: "a.txt" } },
      { tool: "edit_file", args: { path: "a.txt", old_text: "hello", new_text: "goodbye" } },
      { tool: "finish", args: { summary: "Changed the greeting." } },
    ];
    let n = 0;
    const bodies: Array<Record<string, unknown>> = [];
    const server: Server = createServer((req, res) => {
      let b = ""; req.on("data", (c) => (b += c));
      req.on("end", () => {
        bodies.push(JSON.parse(b) as Record<string, unknown>);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ choices: [{ message: { content: JSON.stringify(script[Math.min(n++, script.length - 1)]) }, finish_reason: "stop" }], usage: { prompt_tokens: 5, completion_tokens: 5 } }));
      });
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    cleanup.push(() => server.close());
    const port = (server.address() as { port: number }).port;
    const d = mkdtempSync(join(tmpdir(), "clt-tui-agent-"));
    cleanup.push(() => rmSync(d, { recursive: true, force: true }));
    writeFileSync(join(d, "a.txt"), "hello world\n");
    writeFileSync(join(d, "clutor.config.json"), JSON.stringify({ backend: { kind: "openai", baseUrl: `http://127.0.0.1:${port}/v1`, model: "fake" } }));

    const stdin = new PassThrough() as unknown as NodeJS.ReadStream;
    const stdout = new PassThrough() as unknown as NodeJS.WriteStream;
    (stdout as unknown as { columns: number }).columns = 100;
    let screen = "";
    stdout.on("data", (c) => (screen += String(c)));
    const done = runTui({ io: { out: () => undefined, err: () => undefined, cwd: d }, version: "9.9.9", argv: [], stdin, stdout, color: false, home: d, run: async () => 0 });
    const send = async (s: string, ms = 300) => { stdin.emit("data", s); await new Promise((r) => setTimeout(r, ms)); };
    await new Promise((r) => setTimeout(r, 150));
    await send("\x1b[Z");                       // shift+tab: agent mode
    await send("change the greeting to goodbye\r", 1500);
    expect(readFileSync(join(d, "a.txt"), "utf8")).toBe("hello world\n");   // nothing written yet: waiting for the answer
    await send("\r", 600);                      // "Yes, write the changes"
    await send("/exit\r");
    expect(await done).toBe(0);

    const text = stripAnsi(screen);
    expect(text).toContain("agent · edits need approval");
    expect(text).toContain("● Read(a.txt)");
    expect(text).toContain("Read 1 line");
    expect(text).toContain("● Update(a.txt)");
    expect(text).toContain("- hello");
    expect(text).toContain("+ goodbye");
    expect(text).toContain("Changed the greeting.");
    expect(text).toContain("Apply 1 change to a.txt?");
    expect(text).toContain("applied 1 file");
    expect(readFileSync(join(d, "a.txt"), "utf8")).toBe("goodbye world\n");
    expect(bodies[0]?.["chat_template_kwargs"]).toEqual({ enable_thinking: false });
  });

  it("writes nothing when you say no", async () => {
    const script = [{ tool: "write_file", args: { path: "new.txt", content: "x\n" } }, { tool: "finish", args: { summary: "Made a file." } }];
    let n = 0;
    const server: Server = createServer((req, res) => {
      req.on("data", () => undefined);
      req.on("end", () => { res.writeHead(200, { "content-type": "application/json" }); res.end(JSON.stringify({ choices: [{ message: { content: JSON.stringify(script[Math.min(n++, 1)]) }, finish_reason: "stop" }] })); });
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    cleanup.push(() => server.close());
    const port = (server.address() as { port: number }).port;
    const d = mkdtempSync(join(tmpdir(), "clt-tui-agent2-"));
    cleanup.push(() => rmSync(d, { recursive: true, force: true }));
    writeFileSync(join(d, "clutor.config.json"), JSON.stringify({ backend: { kind: "openai", baseUrl: `http://127.0.0.1:${port}/v1`, model: "fake" } }));
    const stdin = new PassThrough() as unknown as NodeJS.ReadStream;
    const stdout = new PassThrough() as unknown as NodeJS.WriteStream;
    (stdout as unknown as { columns: number }).columns = 100;
    let screen = "";
    stdout.on("data", (c) => (screen += String(c)));
    const done = runTui({ io: { out: () => undefined, err: () => undefined, cwd: d }, version: "9.9.9", argv: [], stdin, stdout, color: false, home: d, run: async () => 0 });
    const send = async (s: string, ms = 300) => { stdin.emit("data", s); await new Promise((r) => setTimeout(r, ms)); };
    await new Promise((r) => setTimeout(r, 150));
    await send("/agent make a file\r", 1500);
    await send("\x1b[B", 200);                  // down to "No, discard them"
    await send("\r", 500);
    await send("/exit\r");
    expect(await done).toBe(0);
    expect(stripAnsi(screen)).toContain("nothing was written");
    expect(existsSync(join(d, "new.txt"))).toBe(false);
  });
});

describe("welcome panel", () => {
  const info = { version: "1.2.3", cwd: "/work/proj", model: "Qwen3.5-9B", detail: "5.6 GB · 8.2k context", recent: ["fix the parser", "explain mmap"] };
  it("is a two-column panel with the version in the border, every row exactly as wide as the panel", async () => {
    const { welcomeLines } = await import("../src/tui/welcome");
    const rows = welcomeLines(plain, 100, info);
    expect(rows[0]).toContain("Clutor v1.2.3");
    expect(rows[0]?.startsWith("╭")).toBe(true);
    expect(rows[rows.length - 1]?.startsWith("╰")).toBe(true);
    for (const r of rows) expect(width(r)).toBe(100);
    const text = rows.join("\n");
    for (const s of ["Welcome back!", "C L U T O R", "Qwen3.5-9B", "Tips for getting started", "shift+tab", "Recent conversations", "· fix the parser"]) expect(text).toContain(s);
  });
  it("folds into one column on a narrow terminal and says when there is no model", async () => {
    const { welcomeLines } = await import("../src/tui/welcome");
    const rows = welcomeLines(plain, 50, { ...info, model: "", recent: [] });
    for (const r of rows) expect(width(r)).toBe(50);
    expect(rows.join("\n")).toContain("Welcome");
    const wide = welcomeLines(plain, 100, { ...info, model: "", recent: [] }).join("\n");
    expect(wide).toContain("no model loaded");
    expect(wide).toContain("nothing yet");
  });
});
