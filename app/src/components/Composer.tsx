import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { PlusMenu } from "./PlusMenu";
import { CloseIcon, SendIcon, StopIcon } from "./icons";

interface Props {
  busy: boolean;
  disabled: boolean;
  placeholder: string;
  thinking: boolean;
  onThinking: (v: boolean) => void;
  browse: boolean;
  onBrowse: (v: boolean) => void;
  onSend: (text: string) => void;
  /** Runs the question as a background job: the reply keeps generating while you do other things. */
  onBackground: (text: string) => void;
  notice?: string | undefined;
  /** The model picker, shown under the text box. */
  picker?: ReactNode;
  /** Puts text in the box (a suggestion); `n` changes each time so the same text can be set again. */
  prefill?: { text: string; n: number } | undefined;
  /** Centered on the empty home screen: a larger box with room to breathe. */
  home?: boolean;
  onStop: () => void;
}

export function Composer({ busy, disabled, placeholder, thinking, onThinking, browse, onBrowse, onSend, onBackground, notice, onStop, picker, prefill, home }: Props) {
  const [text, setText] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);
  const touch = typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;

  useEffect(() => { if (prefill) { setText(prefill.text); area.current?.focus(); } }, [prefill?.n]);
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
  const background = (): void => {
    const t = text.trim();
    if (!t || disabled) return;
    onBackground(t);
    setText("");
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !e.nativeEvent.isComposing) { e.preventDefault(); background(); return; }
    if (e.key === "Enter" && !e.shiftKey && !touch && !e.nativeEvent.isComposing) { e.preventDefault(); send(); }
  };

  return (
    <div className={`composer-wrap${home ? " home" : ""}`}>
      {notice ? <div className="notice" role="status">{notice}</div> : null}
      <div className={`composer${disabled ? " is-disabled" : ""}`}>
        <textarea ref={area} rows={1} value={text} placeholder={placeholder} onChange={(e) => setText(e.target.value)} onKeyDown={onKey} disabled={disabled} enterKeyHint="send" aria-label="Message" />
        <div className="composer-row">
          <PlusMenu items={[
            { id: "think", label: "Think first", hint: "Let thinking models reason before answering (slower, often better)", checked: thinking, onSelect: () => onThinking(!thinking) },
            { id: "browse", label: "Browse the web", hint: "Look things up first; your message goes to a search service", checked: browse, onSelect: () => onBrowse(!browse) },
            { id: "bg", label: "Run in the background", hint: "Keeps generating while you do other things (Ctrl+Enter)", disabled: !text.trim() || disabled, onSelect: background },
          ]} />
          {thinking ? <button type="button" className="pill" onClick={() => onThinking(false)} title="Turn off">Think <CloseIcon /></button> : null}
          {browse ? <button type="button" className="pill" onClick={() => onBrowse(false)} title="Turn off">Browse <CloseIcon /></button> : null}
          <span className="spacer" />
          {picker}
          {busy
            ? <button type="button" className="round stop" onClick={onStop} aria-label="Stop"><StopIcon /></button>
            : <button type="button" className="round send" onClick={send} disabled={!text.trim() || disabled} aria-label="Send"><SendIcon /></button>}
        </div>
      </div>
      {home ? null : <p className="fineprint">Runs on this device with Clutor's engine. Replies can be wrong.</p>}
    </div>
  );
}
