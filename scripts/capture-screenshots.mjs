// Regenerates the screenshots in docs/screenshots from the interface preview (the app's UI running in a browser with sample data: see app/src/bridge/mock.ts).
//
//   cd app && npm run dev            # in one terminal: serves the preview on http://localhost:5173
//   npm i --no-save playwright-core  # a small library (no browser download); needs Chrome, Chromium, Edge or Brave installed
//   BROWSER="/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" node scripts/capture-screenshots.mjs
//
// The preview has no model behind it: its chat reply says so, and its picture is replaced here by a real picture made by Qwen-Image-2.1 Turbo (docs/screenshots/sample-picture.jpg).
import { chromium } from "playwright-core";
import { mkdirSync, readFileSync } from "node:fs";

const url = process.env.PREVIEW ?? "http://localhost:5173";
const out = new URL("../docs/screenshots/", import.meta.url).pathname;
const browser = process.env.BROWSER ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
mkdirSync(out, { recursive: true });
const b = await chromium.launch({ executablePath: browser, headless: true, args: ["--no-sandbox"] });
const page = async () => { const p = await (await b.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 })).newPage(); await p.goto(url, { waitUntil: "networkidle" }); await p.getByText("Skip for now").click(); await p.waitForTimeout(2500); return p; };
const shot = async (p, name) => { await p.waitForTimeout(400); await p.screenshot({ path: `${out}${name}.png` }); console.log("shot", name); };
const message = (p) => p.locator('textarea[aria-label="Message"]');

// chat, the models folder, the model menu, a picture
let p = await page();
await shot(p, "models-folder");
await p.getByRole("button", { name: "Close" }).click(); await p.waitForTimeout(400);
await message(p).fill("How does a mixture-of-experts model run when it is bigger than my memory?");
await message(p).press("Enter");
await p.waitForTimeout(9000);
await shot(p, "chat");
await p.locator(".picker-btn").last().click(); await p.waitForTimeout(500);
await shot(p, "model-menu-auto");
await p.keyboard.press("Escape"); await p.waitForTimeout(300);
await p.getByRole("button", { name: "Image" }).click();
await message(p).fill("a red apple on a wooden table, soft studio lighting, photograph");
await message(p).press("Enter");
await p.waitForTimeout(6000);
const apple = readFileSync(`${out}sample-picture.jpg`).toString("base64");
await p.evaluate((x) => { const im = document.querySelector(".gen-image img"); if (im) im.src = "data:image/jpeg;base64," + x; }, apple);
await shot(p, "image-generation");

// the coding mode with a sample task
p = await page();
await p.getByRole("button", { name: "Close" }).click(); await p.waitForTimeout(400);
await p.locator(".mode-icons button").nth(1).click(); await p.waitForTimeout(700);
await p.getByRole("button", { name: "Choose a project folder" }).click(); await p.waitForTimeout(900);
const task = p.locator('textarea[aria-label="Task"]');
await task.fill("Add an exclamation mark to the greeting and run the tests");
await task.press("Enter");
await p.waitForTimeout(11000);
await shot(p, "coding-mode");
await b.close();
