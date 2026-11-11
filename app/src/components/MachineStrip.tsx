import type { SystemInfo } from "../bridge";
import { fitLabel, type HomeRow } from "../home";

const gb = (b: number): string => `${(b / 2 ** 30).toFixed(b >= 10 * 2 ** 30 ? 0 : 1)} GB`;

/** Home screen: this computer, and what Clutor would use on it for each task. Rows with no model are left out rather than shown empty. */
export function MachineStrip({ system, rows, onPick }: { system: SystemInfo | undefined; rows: HomeRow[]; onPick: () => void }) {
  const shown = rows.filter((r) => r.model);
  return (
    <section className="machine" aria-label="This computer">
      <p className="machine-line">{system ? `${system.os} ${system.arch} · ${Math.round(system.ramBytes / 2 ** 30)} GB memory · ${system.cores} cores` : "Reading this computer…"}</p>
      {shown.length === 0 ? null : (
        <ul className="machine-rows">
          {shown.map((r) => (
            <li key={r.task} title={r.reason}>
              <span className="task">{r.task}</span>
              <span className="mname">{r.model}</span>
              <span className="msize">{gb(r.bytes)}</span>
              <span className={`fit fit-${r.fit}`}>{r.fit === "streams" || r.fit === "tight" ? "⚠ " : r.fit === "memory" ? "✓ " : ""}{fitLabel(r.fit)}</span>
              {r.reason ? <small className="why">Why: {r.reason}</small> : null}
            </li>
          ))}
        </ul>
      )}
      <button type="button" className="linklike" onClick={onPick}>Choose a different model</button>
    </section>
  );
}
