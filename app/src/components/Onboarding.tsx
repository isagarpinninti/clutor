import { useState } from "react";
import { Logo } from "./Logo";

/** The first screen after install: asks what to call you, which is used in greetings and can be changed in Settings. */
export function Onboarding({ onDone }: { onDone: (name: string | undefined) => void }) {
  const [name, setName] = useState("");
  const go = (): void => onDone(name.trim().slice(0, 40) || undefined);
  return (
    <div className="onboard" role="dialog" aria-label="Welcome" data-tauri-drag-region>
      <form className="onboard-card" onSubmit={(e) => { e.preventDefault(); go(); }}>
        <Logo size={56} />
        <h1>Welcome to Clutor</h1>
        <p>Large language models, running on your own machine. Everything you do here stays on this device.</p>
        <label>What should we call you?
          <input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="Your name" maxLength={40} aria-label="Your name" />
        </label>
        <button type="submit" className="btn">Continue</button>
        <button type="button" className="link" onClick={() => onDone(undefined)}>Skip for now</button>
        <small>You can change this any time in Settings.</small>
      </form>
    </div>
  );
}
