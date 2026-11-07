import type { ReactNode } from "react";
import { AlertIcon, InfoIcon } from "./icons";

/** A problem or a heads-up, in one consistent form: an icon, a short title, a plain explanation, and what you can do about it. */
export function Alert({ tone = "error", title, children, action }: { tone?: "error" | "warn" | "info"; title: string; children?: ReactNode; action?: { label: string; onClick: () => void } | undefined }) {
  return (
    <div className={`alert ${tone}`} role={tone === "error" ? "alert" : "status"}>
      <span className="alert-icon" aria-hidden="true">{tone === "info" ? <InfoIcon /> : <AlertIcon />}</span>
      <div className="alert-body">
        <b>{title}</b>
        {children ? <p>{children}</p> : null}
        {action ? <button type="button" className="btn small ghost" onClick={action.onClick}>{action.label}</button> : null}
      </div>
    </div>
  );
}
