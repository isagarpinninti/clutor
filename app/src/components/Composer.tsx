import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { SendIcon, StopIcon } from "./icons";

interface Props {
  busy: boolean;
  disabled: boolean;
  placeholder: string;
  thinking: boolean;
  onThinking: (v: boolean) => void;
  onSend: (text: string) => void;
  onStop: () => void;
}

export function Composer({ busy, disabled, placeholder, thinking, onThinking, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);
  const touch = typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;

  useEffect(() => {
    const a = area.current;
    if (!a) return;
    a.style.height = "auto";
    a.style.height = `${Math.min(a.scrollHeight, 220)}px`;
  }, [text]);

  const send = (): void => {
    const t = text.trim();
    if (!t || busy || disabled) return;
    onSend(t);
    setText("");
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === "Enter" && !e.shiftKey && !touch && !e.nativeEvent.isComposing) { e.preventDefault(); send(); }
  };

  return (
    <div className="composer-wrap">
      <div className={`composer${disabled ? " is-disabled" : ""}`}>
        <textarea ref={area} rows={1} value={text} placeholder={placeholder} onChange={(e) => setText(e.target.value)} onKeyDown={onKey} disabled={disabled} enterKeyHint="send" aria-label="Message" />
        <div className="composer-row">
          <button type="button" className={`chip${thinking ? " on" : ""}`} onClick={() => onThinking(!thinking)} title="Let thinking models reason first (slower, often better)" aria-pressed={thinking}>Think</button>
          <span className="spacer" />
          {busy
            ? <button type="button" className="round stop" onClick={onStop} aria-label="Stop"><StopIcon /></button>
            : <button type="button" className="round send" onClick={send} disabled={!text.trim() || disabled} aria-label="Send"><SendIcon /></button>}
        </div>
      </div>
      <p className="fineprint">Runs on this device with Clutor's engine. Replies can be wrong.</p>
    </div>
  );
}
