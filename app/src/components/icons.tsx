import type { ReactNode } from "react";

const base = { width: 18, height: 18, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" } as const;
const I = (children: ReactNode) => <svg {...base} aria-hidden="true">{children}</svg>;

export const PlusIcon = () => I(<><path d="M12 5v14" /><path d="M5 12h14" /></>);
export const SendIcon = () => I(<><path d="M12 19V5" /><path d="m5 12 7-7 7 7" /></>);
export const StopIcon = () => <svg {...base} fill="currentColor" stroke="none" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2.5" /></svg>;
export const GearIcon = () => I(<><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z" /></>);
export const TrashIcon = () => I(<><path d="M3 6h18" /><path d="M8 6V4h8v2" /><path d="m19 6-1 14H6L5 6" /></>);
export const CopyIcon = () => I(<><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V6a2 2 0 0 1 2-2h9" /></>);
export const ChipIcon = () => I(<><rect x="7" y="7" width="10" height="10" rx="2" /><path d="M10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4" /></>);
export const MenuIcon = () => I(<><path d="M4 7h16" /><path d="M4 12h16" /><path d="M4 17h16" /></>);
export const DownloadIcon = () => I(<><path d="M12 4v11" /><path d="m7 11 5 5 5-5" /><path d="M5 20h14" /></>);
export const FolderIcon = () => I(<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z" />);
export const SparkIcon = () => <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2.5 13.9 9l6.6 1.9-6.6 1.9L12 19.5l-1.9-6.7L3.5 10.9 10.1 9z" /></svg>;
export const ClockIcon = () => I(<><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>);
