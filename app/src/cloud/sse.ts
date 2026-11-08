/** Reads the text of a streamed response (server-sent events) as it arrives in pieces, and gives back each `data:` payload with the event name that came before it. */
export class SseParser {
  private buf = "";
  push(text: string): Array<{ event: string; data: string }> {
    this.buf += text.replace(/\r\n/g, "\n");
    const out: Array<{ event: string; data: string }> = [];
    let at: number;
    while ((at = this.buf.indexOf("\n\n")) >= 0) {
      const block = this.buf.slice(0, at);
      this.buf = this.buf.slice(at + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length) out.push({ event, data: data.join("\n") });
    }
    return out;
  }
}
