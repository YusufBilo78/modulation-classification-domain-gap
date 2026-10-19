"""STUDY_simple.md -> STUDY_simple.html, one self-contained page for a phone.

    python deck/study_html.py

Every slide is a collapsible card; the English script is set apart so it can
be read aloud; a row of numbered chips jumps between slides.
"""
import html
import pathlib
import re

import markdown

HERE = pathlib.Path(__file__).parent
import sys

# python deck/study_html.py [STUDY_x.md | READ_x.md]  -> .html next to it (default: STUDY_simple.md)
SRC = HERE / (sys.argv[1] if len(sys.argv) > 1 else "STUDY_simple.md")
OUT = SRC.with_suffix(".html")
# a file named *_en.md gets an English page: labels, buttons and card heads
EN = SRC.stem.endswith("_en")

CSS = """
:root{--bg:#F6F8FB;--card:#FFFFFF;--ink:#1E293B;--muted:#5B6B7F;--line:#E2E8F0;
--accent:#F08A24;--navy:#0F1B2D;--say:#FFF6EC;--sayline:#F3C899;--chip:#EEF2F6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0B1220;--card:#131C2E;
--ink:#E5EAF2;--muted:#94A3B8;--line:#24324A;--navy:#E5EAF2;--say:#2A2116;--sayline:#7A5326;--chip:#1C2740}}
:root[data-theme="dark"]{--bg:#0B1220;--card:#131C2E;--ink:#E5EAF2;--muted:#94A3B8;--line:#24324A;
--navy:#E5EAF2;--say:#2A2116;--sayline:#7A5326;--chip:#1C2740}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
-webkit-text-size-adjust:100%}
main{max-width:760px;margin:0 auto;padding:12px 16px 80px;overflow-wrap:anywhere}
h1{font-size:1.35rem;line-height:1.3;margin:12px 0 4px;color:var(--navy)}
nav{position:sticky;top:0;z-index:5;background:var(--bg);padding:8px 16px;
display:flex;gap:6px;overflow-x:auto;scrollbar-width:none;border-bottom:1px solid var(--line)}
nav::-webkit-scrollbar{display:none}
nav a{flex:0 0 auto;min-width:36px;height:36px;display:flex;align-items:center;justify-content:center;
padding:0 10px;border-radius:18px;background:var(--chip);color:var(--ink);text-decoration:none;font-weight:600;font-size:.9rem}
nav a:last-child{margin-right:16px}
details{background:var(--card);border:1px solid var(--line);border-radius:14px;margin:14px 0;scroll-margin-top:60px}
summary{cursor:pointer;list-style:none;padding:14px 16px;font-weight:700;font-size:1.05rem;display:flex;gap:10px;align-items:baseline}
summary::-webkit-details-marker{display:none}
summary .n{flex:0 0 auto;color:#fff;background:var(--accent);border-radius:10px;padding:0 8px;font-size:.85rem}
summary::after{content:"+";margin-left:auto;color:var(--muted);font-weight:400}
details[open] summary::after{content:"−"}
.body{padding:0 16px 14px}
.body p,.body li{overflow-wrap:anywhere}
.body strong{color:var(--navy)}
.body ul{padding-left:20px}
.body li{margin:4px 0}
blockquote{margin:10px 0;padding:10px 14px;background:var(--say);border-left:4px solid var(--sayline);border-radius:8px;font-size:1.02rem}
blockquote p{margin:6px 0}
.tag{display:inline-block;font-size:.75rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:14px 0 2px}
.table{overflow-x:auto;max-width:100%;margin:8px 0}
td code{word-break:break-all}
table{border-collapse:collapse;font-size:.88rem;min-width:100%}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}
.nw{white-space:nowrap}
code{background:var(--chip);border-radius:4px;padding:1px 4px;font-size:.85em;overflow-wrap:anywhere}
.intro{color:var(--muted);font-size:.95rem}
.tools{display:flex;gap:8px;margin:10px 0}
.tools button{flex:1;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:10px;padding:8px;font-size:.9rem}
"""

JS = """
const all=()=>document.querySelectorAll('details');
document.getElementById('open').onclick=()=>all().forEach(d=>d.open=true);
document.getElementById('close').onclick=()=>all().forEach(d=>d.open=false);
document.querySelectorAll('nav a').forEach(a=>a.onclick=()=>{const d=document.querySelector(a.getAttribute('href'));if(d)d.open=true;});
"""


def md(text):
    # python-markdown nests lists at 4 spaces; the study file uses 2
    text = re.sub(r"^( +)(?=[-*] |\d+\. )", lambda m: m[1] * 2, text, flags=re.M)
    out = markdown.markdown(text, extensions=["tables"])
    out = out.replace("<table>", '<div class="table"><table>').replace("</table>", "</table></div>")
    # short cells (numbers, dB levels) must not break mid-number on a phone
    out = re.sub(r"<(td|th)>([^<]{1,14})</\1>", r'<\1 class="nw">\2</\1>', out)
    # the three sub-heads of every slide, shown as small labels
    return re.sub(r"<p><strong>(Slaytta ne var, ne demek|Okuma metni \(EN\)|Olası sorular|Ne diyor|Ne demek|Doğru mu\?|Moshe sorarsa \(EN\)|On the slide|What it means|What it is not|In the paper|If asked|Say it in 30 seconds)</strong></p>",
                  r'<div class="tag">\1</div>', out)


def main():
    text = SRC.read_text()
    title = re.match(r"# (.+)", text).group(1).replace("`", "")
    chunks = re.split(r"^## ", text, flags=re.M)
    intro = md(re.sub(r"^# .+\n", "", chunks[0]).replace("\n---\n", "\n"))
    cards, chips = [], []
    for i, ch in enumerate(chunks[1:]):
        head, _, body = ch.partition("\n")
        body = re.sub(r"\n---\s*$", "", body.strip())
        # "Slayt 3 — …" / "Bölüm 4.2 — …" get a numbered chip, "[x] …" chip x
        m = re.match(r"(?:Slayt|Bölüm|Slide|Section) ([\d.]+) — (.+)", head)
        t = re.match(r"\[(.+?)\] (.+)", head)
        if m:
            key, label, name = f"s{m[1].replace('.', '-')}", m[1], m[2]
        elif t:
            key, label, name = f"x{i}", t[1], t[2]
        else:
            key, label, name = f"x{i}", "★" if i == 0 else "?" if head.startswith(("Sözlük", "Glossary")) else "✓", head
        chips.append(f'<a href="#{key}">{label}</a>')
        cards.append(f'<details id="{key}"{" open" if i == 0 else ""}><summary><span class="n">{label}</span>'
                     f'<span>{html.escape(name)}</span></summary><div class="body">{md(body)}</div></details>')
    lang, label, opn, cls = ("en", "Study notes", "Open all", "Close all") if EN else ("tr", "Sunum çalışma metni", "Hepsini aç", "Hepsini kapat")
    page = (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{label}</title><style>{CSS}</style></head><body>'
            f'<nav>{"".join(chips)}</nav><main><h1>{html.escape(title)}</h1>'
            f'<div class="intro">{intro}</div>'
            f'<div class="tools"><button id="open">{opn}</button><button id="close">{cls}</button></div>'
            f'{"".join(cards)}</main><script>{JS}</script></body></html>')
    OUT.write_text(page)
    print("wrote", OUT, len(page) // 1024, "KB,", len(cards), "cards")


if __name__ == "__main__":
    main()
