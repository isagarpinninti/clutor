import type { Style } from "./ansi";

/**
 * A small syntax highlighter for code blocks in replies. It works one line at a time (so it can colour a line while it is still
 * arriving) and knows only the shapes every language shares: comments, strings, numbers, keywords, calls and capitalised names.
 */

type Family = "c" | "hash" | "json" | "sql" | "none";

const KEYWORDS: Record<string, string> = {
  js: "const let var function return if else for while do switch case break continue new class extends import export from default async await try catch finally throw typeof instanceof in of this super null undefined true false void yield static interface type enum implements public private protected readonly as",
  py: "def class return if elif else for while in not and or is import from as with try except finally raise pass break continue lambda yield None True False global nonlocal async await del assert self",
  sh: "if then else elif fi for while do done case esac function in return export local echo cd exit set unset source alias",
  c: "int long short char float double void unsigned signed const static struct union enum typedef sizeof return if else for while do switch case break continue goto extern inline volatile register NULL true false bool size_t",
  rust: "fn let mut const static struct enum impl trait pub use mod crate self Self super match if else for while loop break continue return as in ref move async await dyn where type unsafe true false",
  go: "func var const type struct interface map chan package import return if else for range switch case default break continue go defer select nil true false fallthrough",
  java: "class interface extends implements public private protected static final void int long double float boolean char byte short new return if else for while do switch case break continue try catch finally throw throws import package this super null true false abstract enum instanceof var",
  sql: "select from where and or not insert into values update set delete create table drop alter index join left right inner outer on group by order having limit offset as distinct union null is in like between primary key foreign references",
};

const LANG: Record<string, { fam: Family; kw: string }> = {
  js: { fam: "c", kw: "js" }, javascript: { fam: "c", kw: "js" }, jsx: { fam: "c", kw: "js" }, ts: { fam: "c", kw: "js" }, typescript: { fam: "c", kw: "js" }, tsx: { fam: "c", kw: "js" }, mjs: { fam: "c", kw: "js" },
  py: { fam: "hash", kw: "py" }, python: { fam: "hash", kw: "py" },
  sh: { fam: "hash", kw: "sh" }, bash: { fam: "hash", kw: "sh" }, zsh: { fam: "hash", kw: "sh" }, shell: { fam: "hash", kw: "sh" }, console: { fam: "hash", kw: "sh" },
  c: { fam: "c", kw: "c" }, cpp: { fam: "c", kw: "c" }, "c++": { fam: "c", kw: "c" }, h: { fam: "c", kw: "c" },
  rust: { fam: "c", kw: "rust" }, rs: { fam: "c", kw: "rust" }, go: { fam: "c", kw: "go" }, golang: { fam: "c", kw: "go" },
  java: { fam: "c", kw: "java" }, kotlin: { fam: "c", kw: "java" }, swift: { fam: "c", kw: "java" }, cs: { fam: "c", kw: "java" }, csharp: { fam: "c", kw: "java" },
  json: { fam: "json", kw: "" }, yaml: { fam: "hash", kw: "" }, yml: { fam: "hash", kw: "" }, toml: { fam: "hash", kw: "" },
  sql: { fam: "sql", kw: "sql" },
};

const sets = new Map<string, Set<string>>();
function kwSet(name: string): Set<string> {
  let s = sets.get(name);
  if (!s) { s = new Set((KEYWORDS[name] ?? "").split(" ").filter(Boolean)); sets.set(name, s); }
  return s;
}

/** Colours one line of `lang` code. An unknown language is shown in the plain code colour. */
export function highlightLine(line: string, lang: string, st: Style): string {
  const spec = LANG[lang.toLowerCase()];
  if (!spec || !st.color) return st.code(line);
  const kws = kwSet(spec.kw);
  const ci = spec.fam === "sql";
  let out = "";
  let i = 0;
  while (i < line.length) {
    const c = line[i] as string;
    const rest = line.slice(i);
    if ((spec.fam === "c" && rest.startsWith("//")) || (spec.fam === "hash" && c === "#") || (spec.fam === "sql" && rest.startsWith("--"))) { out += st.tok("com", rest); break; }
    if (spec.fam === "c" && rest.startsWith("/*")) {
      const end = rest.indexOf("*/", 2);
      const cut = end < 0 ? rest.length : end + 2;
      out += st.tok("com", rest.slice(0, cut)); i += cut; continue;
    }
    if (c === '"' || c === "'" || c === "`") {
      let j = i + 1;
      while (j < line.length && line[j] !== c) j += line[j] === "\\" ? 2 : 1;
      const cut = Math.min(j + 1, line.length);
      const text = line.slice(i, cut);
      // In JSON an object key is a string followed by a colon.
      out += spec.fam === "json" && /^\s*:/.test(line.slice(cut)) ? st.tok("fn", text) : st.tok("str", text);
      i = cut; continue;
    }
    const num = /^(0x[0-9a-fA-F_]+|\d[\d_]*(\.\d+)?([eE][+-]?\d+)?)/.exec(rest);
    if (num && !/[\w$]/.test(line[i - 1] ?? "")) { out += st.tok("num", num[0]); i += num[0].length; continue; }
    const id = /^[A-Za-z_$][\w$]*/.exec(rest);
    if (id) {
      const w = id[0];
      const after = line[i + w.length];
      if (kws.has(ci ? w.toLowerCase() : w)) out += st.tok("kw", w);
      else if (after === "(") out += st.tok("fn", w);
      else if (/^[A-Z][a-z]/.test(w) || /^[A-Z]+$/.test(w) && w.length > 1 && spec.fam !== "hash") out += st.tok("ty", w);
      else out += st.code(w);
      i += w.length; continue;
    }
    out += st.code(c); i++;
  }
  return out;
}
