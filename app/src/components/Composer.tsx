import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { fmtBytes } from "../store";
import { MAX_FILES, readAttachment, readImage, type Attachment } from "../attachments";
import { PlusMenu } from "./PlusMenu";
import { CloseIcon, FileIcon, GlobeIcon, SendIcon, StopIcon, ThinkIcon } from "./icons";

interface Props {
  busy: boolean;
  disabled: boolean;
  placeholder: string;
  thinking: boolean;
  onThinking: (v: boolean) => void;
  browse: boolean;
  onBrowse: (v: boolean) => void;
  onSend: (text: string, files: Attachment[]) => void;
  /** The loaded model takes pictures: images can be attached. */
  vision?: boolean | undefined;
  /** Runs the question as a background job: the reply keeps generating while you do other things. */
  onBackground: (text: string, files: Attachment[]) => void;
  notice?: string | undefined;
  /** The model picker, shown under the text box. */
  picker?: ReactNode;
  /** Puts text in the box (a suggestion); `n` changes each time so the same text can be set again. */
  prefill?: { text: string; n: number } | undefined;
  /** Centered on the empty home screen: a larger box with room to breathe. */
  home?: boolean;
  onStop: () => void;
}

export function Composer({ busy, disabled, placeholder, thinking, onThinking, browse, onBrowse, onSend, onBackground, vision, notice, onStop, picker, prefill, home }: Props) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState<Attachment[]>([]);
  const [problem, setProblem] = useState<string>();
  const [dragging, setDragging] = useState(false);
  const picker_ = useRef<HTMLInputElement>(null);
  const area = useRef<HTMLTextAreaElement>(null);

  const add = async (list: FileList | File[] | null): Promise<void> => {
    if (!list) return;
    const incoming = [...list];
    const problems: string[] = [];
    const next = [...files];
    for (const f of incoming) {
      if (next.length >= MAX_FILES) { problems.push(`at most ${MAX_FILES} files`); break; }
      if (next.some((x) => x.name === f.name && x.bytes === f.size)) continue;
      const id = `${Date.now().toString(36)}${next.length}`;
      const r = vision && f.type.startsWith("image/") ? await readImage(f, id) : await readAttachment(f, id);
      if (r.ok) next.push(r.file); else problems.push(`${r.name}: ${r.reason}`);
    }
    setFiles(next);
    setProblem(problems.length ? problems.join("; ") : undefined);
  };
  const touch = typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;

  // A suggestion is applied once. The box is rebuilt when the chat starts, so remember which one it has already seen: otherwise the text comes back after it was sent.
  const seen = useRef(prefill?.n);
  useEffect(() => { if (prefill && prefill.n !== seen.current) { seen.current = prefill.n; setText(prefill.text); area.current?.focus(); } }, [prefill?.n]);
  useEffect(() => {
    const a = area.current;
    if (!a) return;
    a.style.height = "auto";
    a.style.height = `${Math.min(a.scrollHeight, 220)}px`;
  }, [text]);

  const words = (): string => text.trim() || (files.length ? "Please look at the attached file." : "");
  const canSend = (text.trim().length > 0 || files.length > 0) && !disabled;
  const send = (): void => {
    const t = words();
    if (!t || busy || disabled) return;
    onSend(t, files);
    setText(""); setFiles([]); setProblem(undefined);
  };
  const background = (): void => {
    const t = words();
    if (!t || disabled) return;
    onBackground(t, files);
    setText(""); setFiles([]); setProblem(undefined);
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !e.nativeEvent.isComposing) { e.preventDefault(); background(); return; }
    if (e.key === "Enter" && !e.shiftKey && !touch && !e.nativeEvent.isComposing) { e.preventDefault(); send(); }
  };

  return (
    <div className={`composer-wrap${home ? " home" : ""}`}>
      {notice ? <div className="notice" role="status">{notice}</div> : null}
      <div className={`composer${disabled ? " is-disabled" : ""}${dragging ? " dragging" : ""}`}
        onDragOver={(e) => { if (e.dataTransfer.types.includes("Files")) { e.preventDefault(); setDragging(true); } }} onDragLeave={() => setDragging(false)}
        onDrop={(e) => { if (e.dataTransfer.files.length) { e.preventDefault(); setDragging(false); void add(e.dataTransfer.files); } }}
        onPaste={(e) => { if (e.clipboardData.files.length) { e.preventDefault(); void add(e.clipboardData.files); } }}>
        {files.length ? (
          <div className="attach-row">
            {files.map((f) => (
              <span key={f.id} className="attach" title={`${f.name} · ${fmtBytes(f.bytes)}`}>{f.image ? <img className="attach-thumb" src={f.image.thumb} alt="" /> : <FileIcon />}<span className="attach-name">{f.name}</span><small>{fmtBytes(f.bytes)}</small>
                <button type="button" onClick={() => setFiles((l) => l.filter((x) => x.id !== f.id))} aria-label={`Remove ${f.name}`}><CloseIcon /></button></span>
            ))}
          </div>
        ) : null}
        {problem ? <div className="attach-problem" role="alert">{problem}</div> : null}
        <input ref={picker_} type="file" multiple hidden onChange={(e) => { void add(e.target.files); e.target.value = ""; }} />
        <textarea ref={area} rows={1} value={text} placeholder={placeholder} onChange={(e) => setText(e.target.value)} onKeyDown={onKey} disabled={disabled} enterKeyHint="send" aria-label="Message" />
        <div className="composer-row">
          <PlusMenu items={[
            { id: "files", label: "Add files", hint: "Text and code files the model can read", disabled, onSelect: () => picker_.current?.click() },
            { id: "think", label: "Think first", hint: "Let thinking models reason before answering (slower, often better)", checked: thinking, onSelect: () => onThinking(!thinking) },
            { id: "browse", label: "Browse the web", hint: "Look things up first; your message goes to a search service", checked: browse, onSelect: () => onBrowse(!browse) },
            { id: "bg", label: "Run in the background", hint: "Keeps generating while you do other things (Ctrl+Enter)", disabled: !canSend, onSelect: background },
          ]} />
          <button type="button" className={`toggle${thinking ? " on" : ""}`} aria-pressed={thinking} onClick={() => onThinking(!thinking)} title="Let thinking models reason before answering (slower, often better)"><ThinkIcon />Think</button>
          <button type="button" className={`toggle${browse ? " on" : ""}`} aria-pressed={browse} onClick={() => onBrowse(!browse)} title="Look things up on the web first; your message goes to a search service"><GlobeIcon />Browse</button>
          <span className="spacer" />
          {picker}
          {busy
            ? <button type="button" className="round stop" onClick={onStop} aria-label="Stop"><StopIcon /></button>
            : <button type="button" className="round send" onClick={send} disabled={!canSend} aria-label="Send"><SendIcon /></button>}
        </div>
      </div>
      {home ? null : <p className="fineprint">Runs on this device with Clutor's engine. Replies can be wrong.</p>}
    </div>
  );
}
