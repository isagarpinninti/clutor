import { memo, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

function textOf(node: ReactNode): string {
  if (typeof node === "string") return node;
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (node && typeof node === "object" && "props" in node) return textOf((node as { props: { children?: ReactNode } }).props.children);
  return "";
}

function CodeBlock({ children }: { children?: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const lang = (/language-([\w+#-]+)/.exec(String((children as { props?: { className?: string } } | undefined)?.props?.className ?? "")) ?? [])[1] ?? "";
  const copy = (): void => {
    void navigator.clipboard?.writeText(textOf(children).replace(/\n$/, "")).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); });
  };
  return (
    <div className="codeblock">
      <div className="codehead"><span>{lang || "code"}</span><button type="button" onClick={copy}>{copied ? "Copied" : "Copy"}</button></div>
      <pre>{children}</pre>
    </div>
  );
}

/** A model's Markdown reply: GitHub-style tables and task lists, syntax-coloured code with a copy button. Raw HTML in the reply is shown as text, never run. */
export const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[[rehypeHighlight, { detect: false, ignoreMissing: true }]]} components={{ pre: CodeBlock as never, a: ({ children, href }) => <a href={href} target="_blank" rel="noreferrer noopener">{children}</a> }}>
        {text}
      </ReactMarkdown>
    </div>
  );
});
