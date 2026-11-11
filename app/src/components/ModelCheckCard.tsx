import { useEffect, useState } from "react";
import { bridge, type ModelInfo } from "../bridge";
import { checkModel, diskFromBench, type DiskInfo, type ModelCheck } from "../plan";
import { contract, estimates } from "../estimate";
import { fmtBytes, readSpeed } from "../store";

interface Props { model: ModelInfo; ramBytes: number; contextTokens: number }

const tone = { fits: "good", streams: "good", tight: "warn", "too-big": "bad" } as const;

/** Reads the model's header and says whether it will run on this computer, where the memory goes, and how fast the disk could let it go. */
export function ModelCheckCard({ model, ramBytes, contextTokens }: Props) {
  const [check, setCheck] = useState<ModelCheck>();
  const [disk, setDisk] = useState<DiskInfo>();
  const [error, setError] = useState<string>();
  const [measuring, setMeasuring] = useState(false);
  const [warning, setWarning] = useState<string>();

  useEffect(() => {
    let live = true;
    setCheck(undefined); setError(undefined);
    checkModel(bridge.inspect, model.path, { hostRamBytes: ramBytes, contextTokens, disk }).then((c) => { if (live) setCheck(c); }, (e: unknown) => { if (live) setError(e instanceof Error ? e.message : String(e)); });
    return () => { live = false; };
  }, [model.path, ramBytes, contextTokens, disk]);

  const measure = async (): Promise<void> => {
    setMeasuring(true); setWarning(undefined);
    try { const r = diskFromBench(await bridge.inspect.diskBench(model.path, 6)); setDisk(r.disk); setWarning(r.warning); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setMeasuring(false); }
  };

  if (error) return <div className="check-card"><p className="err">{error}</p></div>;
  if (!check) return <div className="check-card"><p className="empty-hint">Reading the model's header…</p></div>;
  const parts = check.plan.components.filter((c) => c.bytes > 0);
  const est = estimates(check, { ...(readSpeed(model.name) ? { measuredTps: readSpeed(model.name) as number } : {}) });
  const ct = contract(check, est);
  return (
    <div className="check-card">
      <p className={`verdict ${tone[check.verdict]}`}>{check.headline}</p>
      <ul className="contract" aria-label="Performance contract">
        <li className={ct.loadable ? "yes" : "no"}><span aria-hidden="true">{ct.loadable ? "✓" : "✗"}</span> Loadable</li>
        <li className={ct.executable ? "yes" : "no"}><span aria-hidden="true">{ct.executable ? "✓" : "✗"}</span> Runnable</li>
        <li className={ct.fast ? "yes" : "no"}><span aria-hidden="true">{ct.fast ? "✓" : "✗"}</span> Fast</li>
      </ul>
      <p className="check-use">{ct.use}</p>
      <p className="check-speed">{check.speed}</p>
      {est ? (
        <div className="check-est" aria-label="Estimated times">
          <p className="check-est-head">Estimated on this computer <small>{est.tps.source === "measured" ? "from this model's measured speed" : "worked out from the file and this machine's memory and disk, not measured"}</small></p>
          <p className="check-est-speed">About {est.tps.lowTps >= 10 ? Math.round(est.tps.lowTps) : est.tps.lowTps.toFixed(1)} to {est.tps.highTps >= 10 ? Math.round(est.tps.highTps) : est.tps.highTps.toFixed(1)} tokens per second</p>
          <ul>{est.lines.map((l) => <li key={l.label}><span>{l.label}</span><b>{l.time}</b></li>)}<li><span>Loading the model</span><b>{est.load}</b></li></ul>
          {est.tps.caution ? <p className="check-note warn">{est.tps.caution}</p> : null}
        </div>
      ) : null}
      <dl className="check-facts">
        <div><dt>Architecture</dt><dd>{check.arch}{check.experts > 0 ? ` · ${check.experts} experts, ${check.expertsUsed} used per token` : " · dense"}</dd></div>
        <div><dt>Layers</dt><dd>{check.layers}</dd></div>
        <div><dt>File</dt><dd>{fmtBytes(check.fileBytes)}</dd></div>
      </dl>
      <table className="check-mem"><tbody>{parts.map((c) => <tr key={c.id}><td>{c.label}</td><td>{fmtBytes(c.bytes)}</td></tr>)}</tbody></table>
      {check.notes.map((n) => <p key={n} className="check-note">{n}</p>)}
      {warning ? <p className="check-note warn">{warning}</p> : null}
      <div className="actions"><button type="button" className="btn small ghost" onClick={() => void measure()} disabled={measuring}>{measuring ? "Measuring the disk…" : disk ? "Measure again" : "Measure this disk (6 s)"}</button></div>
      <p className="check-note">Computed from the file and this computer's memory. Nothing here is a measured model speed.</p>
    </div>
  );
}
