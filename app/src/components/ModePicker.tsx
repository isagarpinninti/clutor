import { useEffect, useRef, useState } from "react";
import { ChevronIcon } from "./icons";

export interface ModeOption<T extends string> { id: T; label: string; hint: string }

interface Props<T extends string> { value: T; options: ModeOption<T>[]; onChange: (v: T) => void; label: string }

/** A small chooser for how the agent behaves (ask first, auto-run checks, plan only), sitting beside the model picker. */
export function ModePicker<T extends string>({ value, options, onChange, label }: Props<T>) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent): void => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent): void => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", away); document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", away); document.removeEventListener("keydown", esc); };
  }, [open]);
  const cur = options.find((o) => o.id === value) ?? options[0];
  return (
    <div className="picker" ref={box}>
      <button type="button" className="picker-btn mode" onClick={() => setOpen((o) => !o)} aria-haspopup="listbox" aria-expanded={open} aria-label={label}>
        <span className="picker-name">{cur?.label}</span><ChevronIcon />
      </button>
      {open ? (
        <div className="pmenu" role="listbox" aria-label={label}>
          {options.map((o) => (
            <button type="button" role="option" aria-selected={o.id === value} key={o.id} className={`menu-item two${o.id === value ? " on" : ""}`} onClick={() => { setOpen(false); onChange(o.id); }}>
              <span className="mi-text"><b>{o.label}</b><small>{o.hint}</small></span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
