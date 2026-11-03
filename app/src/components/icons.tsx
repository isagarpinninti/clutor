import type { ReactNode } from "react";

/** One family: a 24-unit grid, 1.5 stroke, round caps and joins, drawn at 18px. Filled shapes are used only where meaning needs weight (stop, spark). */
const base = { width: 18, height: 18, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.5, strokeLinecap: "round", strokeLinejoin: "round" } as const;
const I = (children: ReactNode) => <svg {...base} aria-hidden="true">{children}</svg>;

export const PlusIcon = () => I(<path d="M12 5.5v13M5.5 12h13" />);
export const SendIcon = () => I(<path d="M12 18.5v-13M6.5 11 12 5.5 17.5 11" />);
export const StopIcon = () => <svg {...base} fill="currentColor" stroke="none" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="2.4" /></svg>;
/** Settings: two sliders, lighter than a cog and legible at small sizes. */
export const GearIcon = () => I(<><path d="M4 8h9M17 8h3M4 16h3M11 16h9" /><circle cx="15" cy="8" r="2" /><circle cx="9" cy="16" r="2" /></>);
export const TrashIcon = () => I(<><path d="M4.5 7h15M9.5 7V5.2c0-.4.3-.7.7-.7h3.6c.4 0 .7.3.7.7V7" /><path d="m6.5 7 .8 11.3c.1.7.6 1.2 1.3 1.2h6.8c.7 0 1.2-.5 1.3-1.2L17.5 7M10 11v5M14 11v5" /></>);
export const CopyIcon = () => I(<><rect x="8.5" y="8.5" width="11" height="11" rx="2.5" /><path d="M15.5 8.5V6.8c0-1.2-1-2.3-2.3-2.3H6.8c-1.2 0-2.3 1-2.3 2.3v6.4c0 1.3 1 2.3 2.3 2.3h1.7" /></>);
export const ChipIcon = () => I(<><rect x="7" y="7" width="10" height="10" rx="2.5" /><path d="M10 3.5V7M14 3.5V7M10 17v3.5M14 17v3.5M3.5 10H7M3.5 14H7M17 10h3.5M17 14h3.5" /></>);
export const MenuIcon = () => I(<path d="M4.5 7h15M4.5 12h15M4.5 17h15" />);
export const DownloadIcon = () => I(<><path d="M12 4.5v10M7.5 10.5l4.5 4.5 4.5-4.5" /><path d="M5 19.5h14" /></>);
export const FolderIcon = () => I(<path d="M3.5 7.5c0-1.1.9-2 2-2h3.2c.5 0 1 .2 1.4.6l1 1.1c.3.3.8.5 1.3.5h5.1c1.1 0 2 .9 2 2v7.8c0 1.1-.9 2-2 2h-12c-1.1 0-2-.9-2-2z" />);
export const SparkIcon = () => <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2.5 13.9 9l6.6 1.9-6.6 1.9L12 19.5l-1.9-6.7L3.5 10.9 10.1 9z" /></svg>;
export const ClockIcon = () => I(<><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 1.8" /></>);
export const SidebarIcon = () => I(<><rect x="3.5" y="4.5" width="17" height="15" rx="3.5" /><path d="M9.5 4.5v15" /></>);
export const SearchIcon = () => I(<><circle cx="11" cy="11" r="6.5" /><path d="m16 16 3.5 3.5" /></>);
export const GaugeIcon = () => I(<><path d="M4.5 17.5a7.5 7.5 0 1 1 15 0" /><path d="m12 17.5 3.5-4.5" /><path d="M12 7.5v1" /></>);
export const ChevronIcon = () => <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="m2.2 3.9 2.8 2.8 2.8-2.8" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>;
export const CheckIcon = () => I(<path d="m5.5 12.5 4.2 4.2 8.8-9.4" />);
export const CloseIcon = () => I(<path d="m6.5 6.5 11 11M17.5 6.5l-11 11" />);
