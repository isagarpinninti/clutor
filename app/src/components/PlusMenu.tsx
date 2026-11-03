import { useEffect, useRef, useState, type ReactNode } from "react";
import { CheckIcon, PlusIcon } from "./icons";

export interface PlusItem { id: string; label: string; hint?: string; checked?: boolean; disabled?: boolean; onSelect: () => void }

/** The "+" button in the message box: switches and one-off actions that would otherwise crowd the row. */
export function PlusMenu({ items, label = "Add", icon, className = "plus-btn", align = "left" }: { items: PlusItem[]; label?: string; icon?: ReactNode; className?: string; align?: "left" | "right" }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent): void => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent): void => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", away); document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", away); document.removeEventListener("keydown", esc); };
  }, [open]);
  return (
    <div className="picker" ref={box}>
      <button type="button" className={`${className} tip`} data-tip={label} onClick={() => setOpen((o) => !o)} aria-haspopup="menu" aria-expanded={open} aria-label={label}>{icon ?? <PlusIcon />}</button>
      {open ? (
        <div className={`pmenu${align === "right" ? " right" : ""}`} role="menu" aria-label={label}>
          {items.map((it) => (
            <button type="button" role="menuitemcheckbox" aria-checked={it.checked ?? false} key={it.id} className="menu-item two" disabled={it.disabled} onClick={() => { setOpen(false); it.onSelect(); }}>
              <span className="mi-text"><b>{it.label}</b>{it.hint ? <small>{it.hint}</small> : null}</span>
              {it.checked ? <span className="mi-check"><CheckIcon /></span> : null}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
