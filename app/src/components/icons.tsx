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
export const PanelIcon = () => I(<><rect x="3.5" y="4.5" width="17" height="15" rx="3.5" /><path d="M14.5 4.5v15" /></>);
export const TerminalIcon = () => I(<><rect x="3.5" y="4.5" width="17" height="15" rx="3" /><path d="m7.5 10 3 2.5-3 2.5M13 15h3.5" /></>);
export const GlobeIcon = () => I(<><circle cx="12" cy="12" r="8.5" /><path d="M3.5 12h17M12 3.5c2.4 2.4 3.6 5.2 3.6 8.5s-1.2 6.1-3.6 8.5c-2.4-2.4-3.6-5.2-3.6-8.5S9.6 5.9 12 3.5Z" /></>);
export const DiffIcon = () => I(<><path d="M8 4.5v11M8 15.5a2.2 2.2 0 1 0 0 .01M8 8a2.2 2.2 0 1 0 0-.01" /><path d="M16 8.5v7M13.5 11 16 8.5 18.5 11" /></>);
export const FilesIcon = () => I(<><path d="M7 3.5h6.5L18 8v10.5c0 .6-.4 1-1 1H7c-.6 0-1-.4-1-1v-14c0-.6.4-1 1-1Z" /><path d="M13.5 3.5V8H18" /></>);
export const ChatIcon = () => I(<path d="M5 6.5c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2v7c0 1.1-.9 2-2 2h-5.5L8 19v-3.5H7c-1.1 0-2-.9-2-2z" />);
export const CodeIcon = () => I(<path d="m9 8-4 4 4 4M15 8l4 4-4 4M13.2 6.5l-2.4 11" />);
export const MoreIcon = () => I(<path d="M6.5 12h.01M12 12h.01M17.5 12h.01" strokeWidth="2.6" />);
export const LaptopIcon = () => I(<><rect x="5" y="6" width="14" height="9.5" rx="1.8" /><path d="M3.5 18.5h17" /></>);
export const PaletteIcon = () => I(<><path d="M12 4.5c-4.4 0-8 3.2-8 7.2 0 3.3 2.6 5.8 5.6 5.8h1.3c.8 0 1.3.6 1.3 1.3 0 .6.4 1 1 1 4 0 6.8-3.1 6.8-7C20 7.6 16.4 4.5 12 4.5Z" /><path d="M8.5 11h.01M11.5 8.5h.01M15 9.5h.01" strokeWidth="2.4" /></>);
export const BellIcon = () => I(<><path d="M6.5 16.5V11a5.5 5.5 0 0 1 11 0v5.5l1.2 1.5H5.3z" /><path d="M10 20.5c.4.7 1.1 1 2 1s1.6-.3 2-1" /></>);
export const ShieldIcon = () => I(<path d="M12 3.8 5.5 6.3v5c0 4 2.7 7 6.5 8.9 3.8-1.9 6.5-4.9 6.5-8.9v-5z" />);
export const PlugIcon = () => I(<><path d="M9 4.5v4M15 4.5v4M7 8.5h10v3.2a5 5 0 0 1-10 0z" /><path d="M12 16.7v3" /></>);
export const InfoIcon = () => I(<><circle cx="12" cy="12" r="8.5" /><path d="M12 11v5M12 8h.01" strokeWidth="1.8" /></>);
export const PaperclipIcon = () => I(<path d="m18.5 11-6.2 6.2a4 4 0 0 1-5.7-5.7l6.5-6.5a2.7 2.7 0 0 1 3.8 3.8l-6.5 6.5a1.3 1.3 0 0 1-1.9-1.9l5.9-5.9" />);
export const FileIcon = () => I(<><path d="M7 3.5h6.5L18 8v10.5c0 .6-.4 1-1 1H7c-.6 0-1-.4-1-1v-14c0-.6.4-1 1-1Z" /><path d="M13.5 3.5V8H18" /></>);
export const BackIcon = () => I(<path d="m14.5 6.5-5.5 5.5 5.5 5.5" />);
export const ForwardIcon = () => I(<path d="m9.5 6.5 5.5 5.5-5.5 5.5" />);
export const UserIcon = () => I(<><circle cx="12" cy="8.5" r="3.5" /><path d="M5 19c.6-3.3 3.4-5 7-5s6.4 1.7 7 5" /></>);
export const PhoneIcon = () => I(<><rect x="7" y="3.5" width="10" height="17" rx="2.6" /><path d="M10.5 17.5h3" /></>);

/** Think: an atom, orbits around a nucleus. */
export const ThinkIcon = () => I(<><ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(60 12 12)" /><ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(-60 12 12)" /><circle cx="12" cy="12" r="1.3" fill="currentColor" stroke="none" /></>);

export const EditIcon = () => I(<><path d="M4.5 19.5h4l10-10a2.1 2.1 0 0 0-3-3l-10 10z" /><path d="m13.5 7.5 3 3" /></>);
export const RedoIcon = () => I(<><path d="M19 12a7 7 0 1 1-2.2-5.1" /><path d="M19 4.5v4h-4" /></>);

export const AlertIcon = () => I(<><circle cx="12" cy="12" r="8.5" /><path d="M12 7.8v5" /><path d="M12 16.2h.01" strokeWidth="2" /></>);

/** Progress: a checklist. */
export const ProgressIcon = () => I(<><path d="M9.5 7h9M9.5 12h9M9.5 17h9" /><path d="m4.8 7.2 1 1 1.8-2M4.8 12.2l1 1 1.8-2M4.8 17.2l1 1 1.8-2" /></>);
