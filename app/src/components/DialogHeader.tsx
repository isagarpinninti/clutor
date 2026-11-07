import type { ReactNode } from "react";
import { CloseIcon } from "./icons";

/** The top of every dialog: an icon tile, the title with a line saying what the dialog is for, and a close button. */
export function DialogHeader({ icon, title, subtitle, onClose, settings }: { icon: ReactNode; title: string; subtitle: string; onClose: () => void; settings?: boolean }) {
  return (
    <header className={`dhead${settings ? " shead" : ""}`}>
      <span className="dhead-icon" aria-hidden="true">{icon}</span>
      <div className="dhead-text"><h2>{title}</h2><small>{subtitle}</small></div>
      <button type="button" className="icon-btn dhead-close" onClick={onClose} aria-label="Close"><CloseIcon /></button>
    </header>
  );
}
