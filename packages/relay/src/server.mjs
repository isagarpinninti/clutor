import { createServer as createHttp } from "node:http";
import { createServer as createHttps } from "node:https";
import { readFileSync } from "node:fs";
import { timingSafeEqual, createHash } from "node:crypto";
import { WebSocketServer } from "ws";

/**
 * The relay. A computer running Clutor keeps one connection open to it; a phone connects to the same address and is paired through to that computer. Everything either side
 * sends after the first message is encrypted end to end (see app/src-tauri/src/remote/secure.rs), so the relay only moves bytes: it cannot read, change or forge them. The
 * token only decides who may use this relay; it protects your server, not your data, and should travel over TLS.
 *
 * First message on every connection (text): {"v":1,"role":"host"|"client","host":"<64 hex characters>","token":"..."}. The reply is {"ok":true}, or the connection is closed with
 *   4400 bad request, 4401 wrong token, 4404 the computer is not connected, 4409 too many phones, 4429 too many tries.
 * After that, binary frames. From a phone to its computer the relay prepends the phone's number (4 bytes, big-endian); the computer starts each frame it sends with the number
 * of the phone it is for. The computer is told about phones coming and going with text messages {"t":"client_open","id":N} and {"t":"client_close","id":N}, and can send
 * {"t":"kick","id":N} to drop one.
 */

const HOST_ID = /^[0-9a-f]{64}$/;
const FIRST_MESSAGE_MS = 10_000;

const same = (a, b) => {
  const x = createHash("sha256").update(a).digest(), y = createHash("sha256").update(b).digest();
  return timingSafeEqual(x, y);
};

export function startRelay({ token, port = 8787, bind = "0.0.0.0", maxClientsPerHost = 8, maxFrameBytes = 1_000_000, tls, log = () => {} } = {}) {
  if (!token || token.length < 16) throw new Error("the relay needs a token of at least 16 characters");
  const server = tls ? createHttps({ cert: readFileSync(tls.cert), key: readFileSync(tls.key) }) : createHttp();
  server.on("request", (_req, res) => { res.writeHead(200, { "content-type": "text/plain" }); res.end("clutor relay\n"); });
  const wss = new WebSocketServer({ server, maxPayload: maxFrameBytes });

  /** host id -> { socket, clients: Map<number, socket>, next } */
  const hosts = new Map();
  /** ip -> { fails, until } : a client that keeps sending a wrong token is slowed down. */
  const tries = new Map();

  const tooMany = (ip) => { const t = tries.get(ip); return !!t && t.fails >= 5 && Date.now() < t.until; };
  const failed = (ip) => { const t = tries.get(ip) ?? { fails: 0, until: 0 }; t.fails++; t.until = Date.now() + 60_000; tries.set(ip, t); };
  const sweep = setInterval(() => { for (const [ip, t] of tries) if (Date.now() > t.until) tries.delete(ip); }, 30_000);
  sweep.unref();

  const alive = new WeakSet();
  const pings = setInterval(() => {
    for (const ws of wss.clients) { if (!alive.has(ws) && ws.pinged) { ws.terminate(); continue; } alive.delete(ws); ws.pinged = true; ws.ping(); }
  }, 30_000);
  pings.unref();

  wss.on("connection", (ws, req) => {
    const ip = req.socket.remoteAddress ?? "?";
    ws.on("pong", () => alive.add(ws));
    ws.on("error", () => {});
    const timer = setTimeout(() => ws.close(4400, "no first message"), FIRST_MESSAGE_MS);
    let role = null, hostId = null, clientId = null;

    ws.once("message", (data, isBinary) => {
      clearTimeout(timer);
      if (tooMany(ip)) return ws.close(4429, "too many tries");
      let hello;
      try { if (isBinary) throw new Error("binary"); hello = JSON.parse(data.toString()); } catch { return ws.close(4400, "bad request"); }
      if (hello?.v !== 1 || (hello.role !== "host" && hello.role !== "client") || typeof hello.host !== "string" || !HOST_ID.test(hello.host) || typeof hello.token !== "string") return ws.close(4400, "bad request");
      if (!same(hello.token, token)) { failed(ip); return ws.close(4401, "wrong token"); }
      role = hello.role; hostId = hello.host;

      if (role === "host") {
        const old = hosts.get(hostId);
        if (old) { for (const c of old.clients.values()) c.close(4404, "the computer reconnected"); old.socket.close(4409, "replaced"); }
        hosts.set(hostId, { socket: ws, clients: new Map(), next: 1 });
        log(`host ${hostId.slice(0, 8)} connected`);
      } else {
        const h = hosts.get(hostId);
        if (!h || h.socket.readyState !== 1) return ws.close(4404, "computer not connected");
        if (h.clients.size >= maxClientsPerHost) return ws.close(4409, "too many phones");
        clientId = h.next++ >>> 0;
        h.clients.set(clientId, ws);
        h.socket.send(JSON.stringify({ t: "client_open", id: clientId }));
      }
      ws.send(JSON.stringify({ ok: true, ...(clientId ? { id: clientId } : {}) }));

      ws.on("message", (frame, binary) => {
        if (!binary) {
          if (role !== "host") return;
          try { const m = JSON.parse(frame.toString()); if (m.t === "kick") hosts.get(hostId)?.clients.get(m.id)?.close(4404, "removed by the computer"); } catch { /* ignore */ }
          return;
        }
        const h = hosts.get(hostId);
        if (!h) return;
        if (role === "client") {
          if (h.socket.readyState !== 1) return;
          const head = Buffer.alloc(4); head.writeUInt32BE(clientId);
          h.socket.send(Buffer.concat([head, frame]), { binary: true });
        } else {
          if (frame.length < 4) return;
          h.clients.get(frame.readUInt32BE(0))?.send(frame.subarray(4), { binary: true });
        }
      });
    });

    ws.on("close", () => {
      clearTimeout(timer);
      if (!role) return;
      const h = hosts.get(hostId);
      if (!h) return;
      if (role === "host" && h.socket === ws) {
        for (const c of h.clients.values()) c.close(4404, "the computer went offline");
        hosts.delete(hostId);
        log(`host ${hostId.slice(0, 8)} left`);
      } else if (role === "client" && h.clients.get(clientId) === ws) {
        h.clients.delete(clientId);
        if (h.socket.readyState === 1) h.socket.send(JSON.stringify({ t: "client_close", id: clientId }));
      }
    });
  });

  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, bind, () => {
      const actual = server.address().port;
      resolve({
        port: actual,
        hostCount: () => hosts.size,
        close: () => new Promise((done) => { clearInterval(pings); clearInterval(sweep); for (const ws of wss.clients) ws.terminate(); wss.close(() => server.close(() => done())); }),
      });
    });
  });
}
