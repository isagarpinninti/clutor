import { useState } from "react";
import { Logo } from "./Logo";
import { firstRun, fitLabel, type HomeRow } from "../home";
import type { SystemInfo } from "../bridge";

interface Props { onDone: (name: string | undefined, chooseModel: boolean) => void; system: SystemInfo | undefined; rows: HomeRow[]; files: number; scanning: boolean }

/** The first screens after install: what to call you (used in greetings, changeable in Settings), then what Clutor found on this computer and which model it would start with. */
export function Onboarding({ onDone, system, rows, files, scanning }: Props) {
  const [name, setName] = useState("");
  const [step, setStep] = useState<1 | 2>(1);
  const who = (): string | undefined => name.trim().slice(0, 40) || undefined;
  if (step === 1) {
    return (
      <div className="onboard" role="dialog" aria-label="Welcome" data-tauri-drag-region>
        <form className="onboard-card" onSubmit={(e) => { e.preventDefault(); setStep(2); }}>
          <Logo size={56} />
          <h1>Welcome to Clutor</h1>
          <p>Run AI models that do not fit in your memory, on your own machine. Everything you do here stays on this device.</p>
          <label>What should we call you?
            <input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="Your name" maxLength={40} aria-label="Your name" />
          </label>
          <button type="submit" className="btn">Continue</button>
          <button type="button" className="link" onClick={() => setStep(2)}>Skip for now</button>
          <small>You can change this any time in Settings.</small>
        </form>
      </div>
    );
  }
  const r = firstRun(system, rows, files);
  return (
    <div className="onboard" role="dialog" aria-label="Your computer" data-tauri-drag-region>
      <div className="onboard-card wide">
        <Logo size={56} />
        <h1>Your computer</h1>
        <p>{r.machine}</p>
        <p>{scanning ? "Looking for models…" : r.found}</p>
        {r.recommended ? (
          <section className="onboard-pick" aria-label="Where Clutor would start">
            <small>Clutor would start with</small>
            <b>{r.recommended.model}</b>
            <span className={`fit fit-${r.recommended.fit}`}>{r.recommended.fit === "streams" || r.recommended.fit === "tight" ? "⚠ " : r.recommended.fit === "memory" ? "✓ " : ""}{fitLabel(r.recommended.fit)}</span>
            {r.streamNote ? <p className="note">{r.streamNote}</p> : null}
          </section>
        ) : scanning ? null : <p className="note">Add a model to begin: Clutor will tell you whether it fits, and how fast to expect it, before it loads.</p>}
        <p className="note">Disk speed is not measured yet. It matters for models bigger than memory, and Clutor measures it when you ask it to plan a model.</p>
        <button type="button" className="btn" onClick={() => onDone(who(), !r.recommended)} autoFocus>{r.recommended ? "Start" : "Choose a model"}</button>
      </div>
    </div>
  );
}
