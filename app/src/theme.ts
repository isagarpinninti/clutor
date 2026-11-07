export type ThemePref = "dark" | "light" | "system";
const KEY = "clutor-theme";

/** The theme that is really shown: "system" follows the operating system. */
export function resolveTheme(pref: ThemePref, prefersDark: boolean): "dark" | "light" {
  return pref === "system" ? (prefersDark ? "dark" : "light") : pref;
}

/** Puts the theme on the page, and remembers the choice where index.html can read it before the first paint (so there is no flash of the wrong colours). */
export function applyTheme(pref: ThemePref): () => void {
  const mq = typeof matchMedia === "function" ? matchMedia("(prefers-color-scheme: dark)") : undefined;
  const apply = (): void => {
    const t = resolveTheme(pref, mq ? mq.matches : true);
    const root = document.documentElement;
    root.dataset["theme"] = t;
    root.style.background = t === "dark" ? "#060607" : "#ffffff";
    document.querySelector('meta[name="color-scheme"]')?.setAttribute("content", t);
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", t === "dark" ? "#060607" : "#ffffff");
  };
  apply();
  try { localStorage.setItem(KEY, pref); } catch { /* blocked storage: the choice is remembered by the app's own settings instead */ }
  mq?.addEventListener?.("change", apply);
  return () => mq?.removeEventListener?.("change", apply);
}
