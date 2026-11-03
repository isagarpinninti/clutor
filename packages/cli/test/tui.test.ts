import { mkdtempSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { afterEach, describe, expect, it } from "vitest";
import { Region, fit, makeStyle, stripAnsi, width, wrap } from "../src/tui/ansi";
import { Editor, layout, type Cmd } from "../src/tui/editor";
import { findGguf } from "../src/tui/discover";
import { KeyParser } from "../src/tui/keys";
import { MarkdownStream } from "../src/tui/markdown";
import { spinnerLine } from "../src/tui/spinner";
import { streamChat } from "../src/tui/stream";
import { expandFiles, runTui } from "../src/tui/app";

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
    expect(v.lines[0]).toBe("─".repeat(30));
    expect(v.lines[1]).toBe("❯ hello world");
    expect(v.cursor).toEqual({ row: 1, col: 13 });
    for (const l of v.lines) expect(width(l)).toBeLessThanOrEqual(30);
  });
  it("wraps long input over several rows and tracks the cursor", () => {
    const e = new Editor([], cmds);
    type(e, "x".repeat(50));
    const v = e.view(plain, 24, { left: "", right: "" });
    expect(layout(e.text, 21).length).toBe(3);
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
    expect(text).toContain("Welcome to Clutor");
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
