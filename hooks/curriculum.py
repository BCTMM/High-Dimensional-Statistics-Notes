"""MkDocs hook: builds the sidebar, the daily order and per-note link boxes
from curriculum.yml."""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
_cur = {}


def _load():
    data = yaml.safe_load((ROOT / "curriculum.yml").read_text())
    notes, tracks = {}, []
    for t in data["tracks"]:
        slugs = []
        for n in t["notes"]:
            n = dict(n, track=t["id"], track_title=t["title"], prereqs=n.get("prereqs", []))
            if n["slug"] in notes:
                raise ValueError(f"duplicate slug {n['slug']}")
            notes[n["slug"]] = n
            slugs.append(n["slug"])
        tracks.append(dict(id=t["id"], title=t["title"], slugs=slugs))
    for n in notes.values():
        for p in n["prereqs"]:
            if p not in notes:
                raise ValueError(f"{n['slug']}: unknown prereq {p}")
    return notes, tracks


def daily_order(notes, tracks):
    """Topological order that interleaves tracks.

    Within a track, the next note is the earliest-listed one whose prerequisites
    are all placed. Across tracks, we avoid repeating the previous day's track if
    possible and pick the track with the largest fraction of notes remaining, so
    all tracks progress at a similar pace."""
    placed, order, prev = set(), [], None
    remaining = {t["id"]: list(t["slugs"]) for t in tracks}
    while len(order) < len(notes):
        ready = {}
        for t in tracks:
            nxt = next((s for s in remaining[t["id"]] if all(p in placed for p in notes[s]["prereqs"])), None)
            if nxt:
                ready[t["id"]] = nxt
        if not ready:
            raise ValueError(f"prerequisite cycle among {[s for r in remaining.values() for s in r]}")
        others = [t for t in tracks if t["id"] in ready and t["id"] != prev] or [t for t in tracks if t["id"] in ready]
        best = max(others, key=lambda t: len(remaining[t["id"]]) / len(t["slugs"]))
        slug = ready[best["id"]]
        order.append(slug)
        placed.add(slug)
        remaining[best["id"]].remove(slug)
        prev = best["id"]
    return order


def on_config(config):
    notes, tracks = _load()
    docs = Path(config["docs_dir"])
    published = {s for s in notes if (docs / "notes" / f"{s}.md").exists()}
    order = daily_order(notes, tracks)
    _cur.update(notes=notes, tracks=tracks, published=published, order=order)

    nav = [{"Home": "index.md"}, {"Daily note": "daily.md"}]
    for t in tracks:
        items = [{notes[s]["title"]: f"notes/{s}.md"} for s in t["slugs"] if s in published]
        if items:
            nav.append({t["title"]: items})
    nav += [{"Roadmap": "roadmap.md"}, {"Progress": "progress.md"}]
    config["nav"] = nav
    return config


def _link(slug, from_note=True):
    n = _cur["notes"][slug]
    if slug in _cur["published"]:
        href = f"{slug}.md" if from_note else f"notes/{slug}.md"
        return f"[{n['title']}]({href})"
    return f"{n['title']} *(coming soon)*"


def on_page_markdown(markdown, page, config, files):
    src = page.file.src_uri
    if src == "roadmap.md":
        rows = ["| Day | Note | Track | Status |", "|---:|---|---|---|"]
        for i, s in enumerate(_cur["order"], 1):
            n = _cur["notes"][s]
            title = _link(s, from_note=False).replace(" *(coming soon)*", "")
            star = " ★" if n.get("star") else ""
            status = "✅ available" if s in _cur["published"] else "✍️ planned"
            rows.append(f"| {i} | {title}{star} | {n['track_title']} | {status} |")
        return markdown.replace("<!-- ROADMAP -->", "\n".join(rows))

    if not src.startswith("notes/"):
        return markdown
    slug = Path(src).stem
    n = _cur["notes"].get(slug)
    if n is None:
        return markdown

    # Links to planned-but-unpublished notes become plain text until the note exists.
    def unlink(m):
        target = m.group(2)
        if target in _cur["notes"] and target not in _cur["published"]:
            return f'<span class="soon" title="Coming soon">{m.group(1)}</span>'
        return m.group(0)

    markdown = re.sub(r"\[([^\]]+)\]\(([a-z0-9-]+)\.md(#[^)]*)?\)", unlink, markdown)
    head = f'<div class="note-meta" data-slug="{slug}"></div>\n\n'
    if n["prereqs"]:
        head += '!!! abstract "Builds on"\n' + "".join(f"    - {_link(p)}\n" for p in n["prereqs"]) + "\n"
    # Put the "Builds on" box right after the H1.
    lines = markdown.split("\n")
    h1 = next((i for i, l in enumerate(lines) if l.startswith("# ")), -1)
    markdown = "\n".join(lines[: h1 + 1] + ["", head] + lines[h1 + 1 :])

    leads = [s for s in _cur["order"] if slug in _cur["notes"][s]["prereqs"]]
    if leads:
        markdown += "\n\n---\n\n**Leads to:** " + " · ".join(_link(s) for s in leads) + "\n"
    markdown += '\n\n<div class="read-toggle" data-slug="' + slug + '"></div>\n'
    return markdown


def on_post_build(config):
    out = {
        "order": [s for s in _cur["order"] if s in _cur["published"]],
        "notes": {
            s: {"title": n["title"], "track": n["track_title"], "url": f"notes/{s}/"}
            for s, n in _cur["notes"].items()
            if s in _cur["published"]
        },
    }
    Path(config["site_dir"], "curriculum.json").write_text(json.dumps(out, ensure_ascii=False))
