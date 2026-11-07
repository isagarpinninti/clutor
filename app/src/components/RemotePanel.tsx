import { useCallback, useEffect, useState } from "react";
import QRCode from "qrcode";
import { bridge, type PairingOffer, type RemoteStatus } from "../bridge";
import type { Settings } from "../store";

interface Props { settings: Settings; onChange: (s: Settings) => void }

const ago = (sec: number): string => { if (!sec) return "never"; const d = Math.max(0, Math.round(Date.now() / 1000 - sec)); return d < 90 ? "just now" : d < 5400 ? `${Math.round(d / 60)} min ago` : d < 129600 ? `${Math.round(d / 3600)} h ago` : `${Math.round(d / 86400)} days ago`; };
const folder = (p: string): string => p.split(/[\\/]/).filter(Boolean).pop() ?? p;

function Row({ title, hint, children }: { title: string; hint?: React.ReactNode; children?: React.ReactNode }) {
  return <div className="srow"><div className="srow-text"><b>{title}</b>{hint ? <small>{hint}</small> : null}</div><div className="srow-ctl">{children}</div></div>;
}

/** Settings > Remote: turn phone control on, pair a phone with a QR code, see and remove paired phones, choose where a phone may work, and read what it did. */
export function RemotePanel({ settings, onChange }: Props) {
  const [st, setSt] = useState<RemoteStatus>();
  const [offer, setOffer] = useState<{ o: PairingOffer; img: string; until: number }>();
  const [paired, setPaired] = useState<string>();
  const [log, setLog] = useState<string[]>([]);
  const [error, setError] = useState<string>();
  const remote = settings.remote ?? {};
  const keepAwake = remote.keepAwake !== false;
  const setRemote = (patch: Partial<NonNullable<Settings["remote"]>>): void => onChange({ ...settings, remote: { ...remote, ...patch } });

  const refresh = useCallback(async () => { try { setSt(await bridge.remote.status()); setLog(await bridge.remote.readLog(25)); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } }, []);
  useEffect(() => { void refresh(); const t = setInterval(() => void refresh(), 2000); return () => clearInterval(t); }, [refresh]);

  // A pairing code that was used or has expired goes away; a new phone in the list means it worked.
  const [known, setKnown] = useState<string[]>();
  useEffect(() => {
    if (!st) return;
    const ids = st.devices.map((d) => d.id);
    if (known && offer) { const added = st.devices.find((d) => !known.includes(d.id)); if (added) { setPaired(added.name); setOffer(undefined); } }
    if (offer && (!st.offerOpen || Date.now() > offer.until)) setOffer(undefined);
    setKnown(ids);
  }, [st]);

  const toggle = async (on: boolean): Promise<void> => {
    setError(undefined); setPaired(undefined);
    try { setSt(on ? await bridge.remote.start(keepAwake) : await bridge.remote.stop()); if (!on) setOffer(undefined); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const pair = async (): Promise<void> => {
    setError(undefined); setPaired(undefined);
    try {
      const o = await bridge.remote.pairStart();
      setKnown(st?.devices.map((d) => d.id));
      setOffer({ o, img: await QRCode.toDataURL(o.qr, { margin: 1, width: 280, errorCorrectionLevel: "M" }), until: Date.now() + o.expiresIn * 1000 });
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const chooseProjects = async (): Promise<void> => { const p = await bridge.host.pickFolder(); if (p) setRemote({ projectsRoot: p }); };
  const addAllowed = async (): Promise<void> => { const p = await bridge.host.pickFolder(); if (p && !(remote.allowedRoots ?? []).includes(p)) setRemote({ allowedRoots: [...(remote.allowedRoots ?? []), p] }); };

  return (
    <>
      <Row title="Allow my phone to control this computer" hint="Off by default and off again when you quit. While it is on, only phones you paired can connect, and only over this Wi-Fi network.">
        <button type="button" role="switch" aria-checked={!!st?.on} aria-label="Allow my phone to control this computer" className={`switch${st?.on ? " on" : ""}`} onClick={() => void toggle(!st?.on)}><i /></button>
      </Row>
      {error ? <p className="notice err" role="alert">{error}</p> : null}
      {st?.on ? (
        <div className="remote-live" role="status"><i className="dot running" />Listening on {st.addresses.length ? st.addresses.join(", ") : `port ${st.port}`} · {st.clients} phone{st.clients === 1 ? "" : "s"} connected. Your computer's firewall may ask to allow incoming connections: allow it on private networks only.</div>
      ) : null}

      <Row title="Keep this computer awake while remote is on" hint="A sleeping computer cannot answer your phone."><button type="button" role="switch" aria-checked={keepAwake} aria-label="Keep awake" className={`switch${keepAwake ? " on" : ""}`} onClick={() => { setRemote({ keepAwake: !keepAwake }); if (st?.on) void bridge.remote.setAwake(!keepAwake).then(setSt); }}><i /></button></Row>

      <h4 className="ssub">Where a phone may work</h4>
      <Row title="Projects folder" hint={remote.projectsRoot ? `${remote.projectsRoot}. New projects started from the phone are created here.` : "Not chosen. New projects cannot be created from the phone until you choose one."}>
        <button type="button" className="btn small ghost" onClick={() => void chooseProjects()}>{remote.projectsRoot ? "Change…" : "Choose…"}</button>
      </Row>
      <Row title="Other folders" hint="A phone can open existing projects only inside the projects folder and these. Nothing else on this computer.">
        <button type="button" className="btn small ghost" onClick={() => void addAllowed()}>Add folder…</button>
      </Row>
      {(remote.allowedRoots ?? []).map((p) => <div key={p} className="srow slim"><span title={p}>{folder(p)} <small>{p}</small></span><button type="button" className="btn small ghost" onClick={() => setRemote({ allowedRoots: (remote.allowedRoots ?? []).filter((x) => x !== p) })}>Remove</button></div>)}

      <h4 className="ssub">Phones</h4>
      {paired ? <p className="notice ok" role="status">Paired with {paired}.</p> : null}
      {offer ? (
        <div className="pair-card">
          <img src={offer.img} width={200} height={200} alt="Pairing code: scan it with the Clutor app on your phone" />
          <div><b>Scan this with the Clutor app</b><small>It works once, for two minutes, and only on this network. Never share a picture of it.</small><small>{offer.o.addresses.join(" · ") || "No network address found: connect to Wi-Fi."}</small>
            <button type="button" className="btn small ghost" onClick={() => { void bridge.remote.pairCancel(); setOffer(undefined); }}>Cancel</button></div>
        </div>
      ) : <Row title="Pair a new phone" hint={st?.on ? "Shows a QR code for the Clutor app to scan." : "Turn remote access on first."}><button type="button" className="btn small" disabled={!st?.on} onClick={() => void pair()}>Show QR code</button></Row>}
      {(st?.devices ?? []).length === 0 ? <p className="empty-hint">No phones paired yet.</p> : (st?.devices ?? []).map((d) => (
        <div key={d.id} className="srow slim"><span>{d.name} <small>last connected {ago(d.lastSeen)}</small></span><button type="button" className="btn small danger" onClick={() => void bridge.remote.revoke(d.id).then(setSt)}>Remove</button></div>
      ))}

      <h4 className="ssub">Activity</h4>
      <pre className="remote-log" aria-label="Remote activity log">{log.length ? log.map((l) => { try { const j = JSON.parse(l) as { at?: number; device?: string; what?: string }; return `${j.at ? new Date(j.at).toLocaleTimeString() : ""}  ${j.device ?? ""}  ${j.what ?? l}`; } catch { return l; } }).join("\n") : "Nothing yet. Every request from a phone is listed here."}</pre>
    </>
  );
}
