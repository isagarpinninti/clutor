// Draws the HTML written by scripts/capture-terminal.py as a framed terminal window and saves docs/screenshots/terminal-ui.png and terminal-plan.png.
//   OUT=/tmp python3 scripts/capture-terminal.py tui && OUT=/tmp python3 scripts/capture-terminal.py plan
//   OUT=/tmp BROWSER="/path/to/chrome" node scripts/capture-terminal.mjs     (needs: npm i --no-save playwright-core)
import { chromium } from "playwright-core";
import { readFileSync } from "node:fs";
const dir = process.env.OUT ?? "/tmp";
const docs = new URL("../docs/screenshots/", import.meta.url).pathname;
const b = await chromium.launch({ executablePath: process.env.BROWSER ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: true, args: ["--no-sandbox"] });
const page = (title, cmd, body) => `<!doctype html><meta charset=utf-8><style>
body{margin:0;background:#060608;padding:28px;font-family:-apple-system,Helvetica,sans-serif}
.term{width:900px;background:#0b0b0e;border:1px solid #25252b;border-radius:12px;overflow:hidden;box-shadow:0 12px 40px rgba(0,0,0,.5)}
.bar{display:flex;align-items:center;gap:8px;padding:10px 14px;background:#101014;border-bottom:1px solid #25252b;color:#8a8a94;font-size:12px}
.dot{width:12px;height:12px;border-radius:50%}.r{background:#ff5f56}.y{background:#ffbd2e}.g{background:#27c93f}
pre{margin:0;padding:16px 18px;color:#d7dae0;font:12.5px/1.42 "SF Mono",Menlo,Consolas,monospace;white-space:pre;overflow:hidden}
.cmd{color:#7c8cff}</style><div class=term><div class=bar><i class="dot r"></i><i class="dot y"></i><i class="dot g"></i><span style="margin-left:10px">${title}</span></div><pre>${cmd ? `<span class=cmd>$ ${cmd}</span>\n\n` : ""}${body}</pre></div>`;
for (const [name, title, cmd] of [["terminal-ui", "clutor", ""], ["terminal-plan", "clutor plan", "clutor plan qwen3-30b-a3b --ram 16"]]) {
  // Rows that draw the logo with block characters get line-height 1, as in a real terminal, so the half blocks join up instead of showing stripes.
  // The font used for the picture draws half blocks with gaps; a terminal draws them exactly. So each block character becomes a box of one column and one line, coloured top and bottom.
const px = (l) => l.replace(/<span style="([^"]*)">([^<]*)<\/span>/g, (m, css, text) => {
  if (!/[█▀▄]/.test(text)) return m;
  const fg = /(?:^|;)color:([^;]+)/.exec(css)?.[1] ?? "transparent", bg = /background:([^;]+)/.exec(css)?.[1] ?? "transparent";
  return [...text].map((ch) => {
    const [top, bot] = ch === "█" ? [fg, fg] : ch === "▀" ? [fg, bg] : ch === "▄" ? [bg, fg] : [bg, bg];
    return `<span style="display:inline-block;width:1ch;height:17.5px;vertical-align:top;background:linear-gradient(${top} 50%,${bot} 50%)"></span>`;
  }).join("");
});
const raw = readFileSync(`${dir}/term_${name === "terminal-ui" ? "tui" : "plan"}.html`, "utf8").split("\n");
  const body = raw.map((l, i) => (/[█▀▄]/.test(l) ? `<span style="display:block;line-height:17.5px;height:17.5px">${px(l)}</span>` : i < raw.length - 1 ? `${l}\n` : l)).join("");
  const p = await (await b.newContext({ viewport: { width: 980, height: 900 }, deviceScaleFactor: 2 })).newPage();
  await p.setContent(page(title, cmd, body));
  await p.locator(".term").screenshot({ path: `${docs}${name}.png` });
  console.log("shot", name);
}
await b.close();
