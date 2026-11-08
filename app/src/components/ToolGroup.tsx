import { useState } from "react";
import type { Block } from "../agent/sessions";
import { summarizeTools } from "../agent/sessions";
import { ToolBlock } from "./ToolBlock";
import { Spark } from "./Thinking";
import { CheckIcon, CloseIcon } from "./icons";

type Tool = Extract<Block, { kind: "tool" }>;

/**
 * A run of steps as one line, the way a person would tell it: "Wrote 2 files, ran 3 commands". While one is running the line says what it is doing; click to see every step,
 * and click a step to see its output or its change.
 */
export function ToolGroup({ tools, verbose, onOpenFile }: { tools: Tool[]; verbose: boolean; onOpenFile: (path: string) => void }) {
  const running = tools.find((t) => t.status === "running");
  const failed = tools.some((t) => t.status === "error");
  const [user, setUser] = useState<boolean | undefined>(undefined);
  const open = user ?? (verbose || !!running);
  const live = running ? running.label.replace(/^(Read|Listed|Searched|Wrote|Edited|Deleted|Mapped|Ran|Opened|Used)\b/, (w) => ({ Read: "Reading", Listed: "Listing", Searched: "Searching", Wrote: "Writing", Edited: "Editing", Deleted: "Deleting", Mapped: "Mapping", Ran: "Running", Opened: "Opening", Used: "Using" } as Record<string, string>)[w] ?? w) : undefined;
  return (
    <div className={`tgroup${running ? " running" : ""}${failed ? " failed" : ""}`}>
      <button type="button" className="tgroup-head" aria-expanded={open} onClick={() => setUser(!open)}>
        <span className="tool-state" aria-hidden="true">{running ? <Spark size={20} /> : failed ? <CloseIcon /> : <CheckIcon />}</span>
        <span className="tgroup-text">{live ? <><b className="shimmer">{live}…</b><small>{tools.length > 1 ? ` · ${summarizeTools(tools.filter((t) => t.status !== "running")).toLowerCase()}`.replace(" · planned", "") : ""}</small></> : summarizeTools(tools)}</span>
        <span className={`tool-caret${open ? " open" : ""}`}>›</span>
      </button>
      {open ? <div className="tgroup-body">{tools.map((t) => <ToolBlock key={t.id} block={t} verbose={false} onOpenFile={onOpenFile} />)}</div> : null}
    </div>
  );
}
