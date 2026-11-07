import { useState } from "react";
import type { Block } from "../agent/sessions";
import { Patch } from "./Patch";
import { Spark } from "./Thinking";
import { CheckIcon, CloseIcon } from "./icons";

/** One step of the agent: a line saying what it did, which opens to show the output or the change it made. */
export function ToolBlock({ block, onOpenFile, verbose }: { block: Extract<Block, { kind: "tool" }>; onOpenFile?: (path: string) => void; verbose?: boolean }) {
  const [open, setOpen] = useState(verbose ?? false);
  const detail = block.patch ?? block.output;
  return (
    <div className={`tool ${block.status}`}>
      <button type="button" className="tool-head" onClick={() => setOpen((o) => !o)} aria-expanded={open} disabled={!detail}>
        <span className="tool-state" aria-label={block.status}>{block.status === "running" ? <Spark size={12} /> : block.status === "ok" ? <CheckIcon /> : <CloseIcon />}</span>
        <span className="tool-label">{block.label}</span>
        {block.path && onOpenFile ? <span role="link" tabIndex={0} className="tool-open" onClick={(e) => { e.stopPropagation(); onOpenFile(block.path as string); }} onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); onOpenFile(block.path as string); } }}>open</span> : null}
        {detail ? <span className={`tool-caret${open ? " open" : ""}`}>›</span> : null}
      </button>
      {open && detail ? (block.patch ? <Patch patch={block.patch} /> : <pre className="tool-out">{detail}</pre>) : null}
    </div>
  );
}
