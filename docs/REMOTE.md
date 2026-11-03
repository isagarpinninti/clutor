# Remote control from your phone

The phone app controls Clutor on your computer: you send a task, watch the agent's steps, approve commands and review changes from your phone. Everything runs on the computer (the model, the agent, your files); the phone is a remote.

**Status:** the encrypted channel and the relay server are built and tested. The computer-side host, the pairing screen and the Android app are not built yet.

## How the phone reaches your computer

```
phone  ──►  your relay (a server you run)  ◄──  computer running Clutor
        encrypted end to end; the relay cannot read it
```

* On the same network the phone connects to the computer directly.
* Away from home both sides connect out to a relay that **you host**. A home computer usually cannot be reached from outside (routers and carriers block it), so something with a public address has to meet them. The relay only forwards bytes.

## What protects it

* **Pairing.** The computer shows a QR code holding a one-time secret and its public key. The phone proves it scanned it by mixing the secret into the handshake, and checks that the computer is the one in the code. The secret works once and expires.
* **End-to-end encryption** (Noise `XX`/`XXpsk3`: X25519, ChaCha20-Poly1305, BLAKE2s). The relay, and anyone on the network, sees only ciphertext. Changed, replayed or reordered messages are rejected.
* **Known devices only.** After pairing, the computer accepts only the keys it paired, and the phone only the computer it paired with. Any phone can be revoked from the computer.
* **The relay token** decides who may use your relay. It protects your server, not your data. Put the relay behind TLS, or the token can be read on the way (the data stays encrypted either way).

What the relay operator can learn: that a computer and a phone are connected, when, from which addresses, and how much data moves. Not the content.

## Running the relay

Needs a machine with a public address (a small VPS is enough) and Node.js 20+, or Docker.

```bash
# in packages/relay
RELAY_TOKEN=$(openssl rand -hex 24) node src/bin.mjs          # port 8787
# or with Docker
docker build -t clutor-relay . && docker run -p 8787:8787 -e RELAY_TOKEN=<your token> clutor-relay
```

For TLS either run it behind a proxy such as Caddy (`reverse_proxy localhost:8787` for your domain) or pass `TLS_CERT` and `TLS_KEY` paths. Each connection's first message names a computer by a 64-hex id (derived from its public key) and carries the token. A phone is told "computer not connected" if the computer is offline. At most 8 phones can attach to one computer, frames are capped at 1 MB, and five wrong tokens from one address are followed by a one-minute lockout.

## Rules for remote tasks (to be enforced by the host)

* Remote access is off by default, with a visible indicator and a kill switch on the computer.
* The phone works only inside project folders you allowed. New projects are created inside the projects folder you choose on the computer.
* The phone cannot turn off approvals: running commands and using the network always need Allow.
* The default mode for a remote task is Plan only.
* Every remote action is written to a log you can read on the computer.
* A setting keeps the computer awake while remote access is on.
