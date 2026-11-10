#!/usr/bin/env node
import { startRelay } from "./server.mjs";

const token = process.env.RELAY_TOKEN;
if (!token) {
  console.error("Set RELAY_TOKEN to a long random secret (at least 16 characters), for example:\n  RELAY_TOKEN=$(openssl rand -hex 24) node src/bin.mjs");
  process.exit(1);
}
const tls = process.env.TLS_CERT && process.env.TLS_KEY ? { cert: process.env.TLS_CERT, key: process.env.TLS_KEY } : undefined;
const r = await startRelay({ token, port: Number(process.env.PORT ?? 8787), ...(tls ? { tls } : {}), log: (m) => console.log(new Date().toISOString(), m) });
console.log(`clutor relay listening on port ${r.port}${tls ? " (TLS)" : " (no TLS: put it behind a TLS proxy, or the token can be read on the way)"}`);
process.on("SIGTERM", () => void r.close().then(() => process.exit(0)));
