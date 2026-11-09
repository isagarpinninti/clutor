import { describeLibrary } from "@core/models/library";
import type { Auto } from "../library";
import type { Library } from "@core/models/library";

const row = (label: string, text: string, ok: boolean): JSX.Element => (
  <div className="lib-row"><span className="lib-label">{label}</span><span className={ok ? "" : "lib-none"}>{text}</span></div>
);

/** What Clutor found in the models folder and which model it will use for each kind of task, with the reason. */
export function LibraryCard({ library, auto, scanning, onRescan }: { library: Library; auto: Auto; scanning: boolean; onRescan: () => void }) {
  const summary = describeLibrary(library);
  return (
    <div className="check-card lib-card" aria-label="Models folder">
      <p className="check-est-head">Your models folder <small>{scanning ? "reading the files…" : summary.length ? summary.join(" · ") : "no models found yet"}</small></p>
      {row("Chat", auto.chat.choice ? `${auto.chat.choice.name}` : auto.chat.reason, !!auto.chat.choice)}
      {auto.chat.choice ? <p className="lib-why">{auto.chat.reason}</p> : null}
      {row("Code", auto.code.choice ? `${auto.code.choice.name}` : auto.code.reason, !!auto.code.choice)}
      {row("Pictures", auto.vision.choice ? `${auto.vision.choice.model.name}` : auto.vision.reason, !!auto.vision.choice)}
      {row("Images", auto.image.choice ? `${auto.image.choice.diffusion.name}` : auto.image.reason, !!auto.image.choice)}
      {auto.image.choice ? <p className="lib-why">{auto.image.reason}</p> : null}
      {library.other.length ? (
        <details className="lib-other"><summary>{library.other.length} file{library.other.length === 1 ? "" : "s"} Clutor cannot use</summary>
          <ul>{library.other.slice(0, 20).map((f) => <li key={f.path}><b>{f.name}</b><small>{f.note ?? "not recognised"}</small></li>)}</ul></details>
      ) : null}
      <div className="actions"><button type="button" className="btn small ghost" onClick={onRescan} disabled={scanning}>{scanning ? "Reading…" : "Read the folder again"}</button></div>
    </div>
  );
}
