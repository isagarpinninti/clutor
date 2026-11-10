"""Runs the built `clutor` command line in a pseudo-terminal, lets a terminal emulator (pyte) draw what it printed, and writes that screen as coloured HTML for
scripts/capture-terminal.mjs to turn into a screenshot. Needs `pip install pyte` and `npm run build` first.

    python3 scripts/capture-terminal.py tui    # the interactive screen after one question
    python3 scripts/capture-terminal.py plan   # `clutor plan qwen3-30b-a3b --ram 16`
It runs with an empty home folder so that no conversation history of yours appears in the picture."""
import sys, os, pty, time, select, fcntl, termios, struct, html, tempfile
import pyte
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "packages", "cli", "dist", "bin.js")
WORK = os.path.realpath(tempfile.mkdtemp(prefix="clutor-demo-"))
PROJECT = os.path.join(WORK, "demo-project")   # the working folder shows as ~/demo-project
os.makedirs(PROJECT)
ROWS, COLS = 50, 100
def run(argv, inputs, first=3.0, wait=2.0):
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["TERM"] = "xterm-256color"; os.environ["HOME"] = WORK; os.chdir(PROJECT); os.environ["COLUMNS"] = str(COLS); os.environ["LINES"] = str(ROWS); os.environ["NO_COLOR"] = ""
        os.environ.pop("NO_COLOR", None)
        os.execvp(argv[0], argv)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    screen = pyte.Screen(COLS, ROWS); stream = pyte.ByteStream(screen)
    def pump(t):
        end = time.time() + t
        while time.time() < end:
            r, _, _ = select.select([fd], [], [], 0.2)
            if r:
                try: stream.feed(os.read(fd, 65536))
                except OSError: return
    pump(first)
    for data, w in inputs:
        os.write(fd, data); pump(w)
    try: os.write(fd, b"\x03"); os.write(fd, b"\x03")
    except OSError: pass
    return screen
PAL = {"black":"#1b1d23","red":"#e06c75","green":"#98c379","brown":"#e5c07b","blue":"#61afef","magenta":"#c678dd","cyan":"#56b6c2","white":"#d7dae0","brightblack":"#6b717d","brightred":"#ff7b86","brightgreen":"#a9d98a","brightbrown":"#f0cf8e","brightblue":"#7cc0ff","brightmagenta":"#d98aee","brightcyan":"#6fd0dc","brightwhite":"#ffffff","default":None}
def color(c, bold=False):
    if c in PAL: return PAL[c]
    if isinstance(c, str) and len(c) == 6:
        try: int(c, 16); return "#" + c
        except ValueError: return None
    return None
def to_html(screen):
    lines = []
    for y in range(screen.lines):
        row = screen.buffer[y]; out = []; last = None; cur = ""
        for x in range(screen.columns):
            ch = row[x]
            style = (color(ch.fg), color(ch.bg), ch.bold, ch.italics, ch.reverse)
            if style != last:
                if cur: out.append((last, cur))
                last, cur = style, ""
            cur += ch.data
        out.append((last, cur))
        s = ""
        for st, text in out:
            fg, bg, bold, it, rev = st
            if rev: fg, bg = bg or "#1b1d23", fg or "#d7dae0"
            css = (f"color:{fg};" if fg else "") + (f"background:{bg};" if bg else "") + ("font-weight:700;" if bold else "") + ("font-style:italic;" if it else "")
            s += f'<span style="{css}">{html.escape(text)}</span>' if css else html.escape(text)
        lines.append(s.rstrip())
    while lines and not lines[-1].strip(): lines.pop()
    return "\n".join(lines)
if __name__ == "__main__":
    which = sys.argv[1]
    if which == "tui":
        scr = run(["node", CLI], [(b"Give me three tips for writing clear commit messages.\r", 40.0)])
    else:
        scr = run(["node", CLI, "plan", "qwen3-30b-a3b", "--ram", "16"], [], first=3.0)
    open(os.path.join(os.environ.get("OUT", tempfile.gettempdir()), f"term_{which}.html"), "w").write(to_html(scr))
    print(which, "lines", len(to_html(scr).splitlines()))
