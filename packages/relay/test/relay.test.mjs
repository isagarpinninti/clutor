import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { WebSocket } from "ws";
import { startRelay } from "../src/server.mjs";

const TOKEN = "t".repeat(24);
const HOST = "a".repeat(64);
let relay;
before(async () => { relay = await startRelay({ token: TOKEN, port: 0, bind: "127.0.0.1", maxClientsPerHost: 2, maxFrameBytes: 4096 }); });
after(async () => { await relay.close(); });

const url = () => `ws://127.0.0.1:${relay.port}`;
/** Opens a connection, sends the first message, and resolves with the reply (or the close code). */
function open(hello) {
  return new Promise((resolve) => {
    const ws = new WebSocket(url());
    const texts = [], bins = [];
    ws.on("message", (d, bin) => (bin ? bins.push(Buffer.from(d)) : texts.push(JSON.parse(d.toString()))));
    ws.on("open", () => ws.send(JSON.stringify({ v: 1, ...hello })));
    ws.on("close", (code) => resolve({ ws, code, texts, bins, closed: true }));
    const check = setInterval(() => { if (texts.length) { clearInterval(check); resolve({ ws, texts, bins, closed: false }); } }, 5);
    ws.on("close", () => clearInterval(check));
  });
}
const until = async (f) => { for (let i = 0; i < 200; i++) { if (f()) return; await new Promise((r) => setTimeout(r, 5)); } throw new Error("timed out"); };
const host = () => open({ role: "host", host: HOST, token: TOKEN });
const client = () => open({ role: "client", host: HOST, token: TOKEN });

test("a wrong token is refused", async () => {
  const r = await open({ role: "host", host: HOST, token: "x".repeat(24) });
  assert.equal(r.code, 4401);
});

test("a malformed hello is refused", async () => {
  assert.equal((await open({ role: "host", host: "short", token: TOKEN })).code, 4400);
  assert.equal((await open({ role: "nobody", host: HOST, token: TOKEN })).code, 4400);
});

test("a phone cannot connect while its computer is offline", async () => {
  assert.equal((await client()).code, 4404);
});

test("frames are routed between a computer and each phone, and phones cannot see each other", async () => {
  const h = await host();
  const a = await client(), b = await client();
  await until(() => h.texts.filter((t) => t.t === "client_open").length === 2);
  const [ida, idb] = h.texts.filter((t) => t.t === "client_open").map((t) => t.id);
  assert.notEqual(ida, idb);

  a.ws.send(Buffer.from("from a"));
  b.ws.send(Buffer.from("from b"));
  await until(() => h.bins.length === 2);
  const got = h.bins.map((f) => [f.readUInt32BE(0), f.subarray(4).toString()]);
  assert.deepEqual(got.sort(), [[ida, "from a"], [idb, "from b"]].sort());

  const to = (id, text) => { const head = Buffer.alloc(4); head.writeUInt32BE(id); h.ws.send(Buffer.concat([head, Buffer.from(text)])); };
  to(ida, "for a"); to(idb, "for b");
  await until(() => a.bins.length === 1 && b.bins.length === 1);
  assert.equal(a.bins[0].toString(), "for a");
  assert.equal(b.bins[0].toString(), "for b");
  assert.equal(a.bins.length + b.bins.length, 2);

  b.ws.close();
  await until(() => h.texts.some((t) => t.t === "client_close" && t.id === idb));
  a.ws.close(); h.ws.close();
  await until(() => relay.hostCount() === 0);
});

test("the computer can drop a phone, and a phone is dropped when the computer leaves", async () => {
  const h = await host();
  const a = await client();
  await until(() => h.texts.some((t) => t.t === "client_open"));
  const id = h.texts.find((t) => t.t === "client_open").id;
  const closed = new Promise((r) => a.ws.on("close", (c) => r(c)));
  h.ws.send(JSON.stringify({ t: "kick", id }));
  assert.equal(await closed, 4404);
  const c = await client();
  const gone = new Promise((r) => c.ws.on("close", (code) => r(code)));
  h.ws.close();
  assert.equal(await gone, 4404);
  await until(() => relay.hostCount() === 0);
});

test("only a limited number of phones can connect to one computer", async () => {
  const h = await host();
  const a = await client(), b = await client();
  const c = await client();
  assert.equal(c.code, 4409);
  a.ws.close(); b.ws.close(); h.ws.close();
  await until(() => relay.hostCount() === 0);
});

test("a computer that reconnects replaces the old connection", async () => {
  const h1 = await host();
  const closed = new Promise((r) => h1.ws.on("close", (c) => r(c)));
  const h2 = await host();
  assert.equal(await closed, 4409);
  assert.equal(relay.hostCount(), 1);
  h2.ws.close();
  await until(() => relay.hostCount() === 0);
});

test("an oversized frame closes the connection", async () => {
  const h = await host();
  const closed = new Promise((r) => h.ws.on("close", (c) => r(c)));
  h.ws.send(Buffer.alloc(5000));
  assert.equal(await closed, 1009);
});

test("repeated wrong tokens are slowed down", async () => {
  const r2 = await startRelay({ token: TOKEN, port: 0, bind: "127.0.0.1" });
  const go = () => new Promise((resolve) => { const ws = new WebSocket(`ws://127.0.0.1:${r2.port}`); ws.on("open", () => ws.send(JSON.stringify({ v: 1, role: "host", host: HOST, token: "y".repeat(24) }))); ws.on("close", (c) => resolve(c)); });
  const codes = [];
  for (let i = 0; i < 7; i++) codes.push(await go());
  assert.deepEqual(codes.slice(0, 5), [4401, 4401, 4401, 4401, 4401]);
  assert.equal(codes[5], 4429);
  await r2.close();
});

test("a relay will not start with a weak token", () => {
  assert.throws(() => startRelay({ token: "short" }), /token/);
});
