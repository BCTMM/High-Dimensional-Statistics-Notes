// Daily note logic. All progress lives in this browser's localStorage.
//
// - "Daily note" picks the first note (in curriculum order) not yet seen, and
//   keeps returning that same note for the rest of the calendar day.
// - Opening any note (daily or from the sidebar) marks it seen.
// - When every published note has been seen, a new cycle starts.

(() => {
  const KEY = "daily-concepts:v1";

  const load = () => {
    try {
      const s = JSON.parse(localStorage.getItem(KEY));
      if (s && Array.isArray(s.seen)) return s;
    } catch (e) {}
    return { seen: [], today: null, cycle: 1 };
  };
  const save = (s) => {
    try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) {}
  };
  const todayStr = () => new Date().toLocaleDateString("en-CA"); // YYYY-MM-DD, local time

  const siteRoot = () => {
    try {
      const cfg = JSON.parse(document.getElementById("__config").textContent);
      return new URL(cfg.base.replace(/\/?$/, "/"), location.href);
    } catch (e) {
      return new URL("/", location.href);
    }
  };

  let curriculumPromise = null;
  const curriculum = () => {
    if (!curriculumPromise) {
      curriculumPromise = fetch(new URL("curriculum.json", siteRoot())).then((r) => r.json());
    }
    return curriculumPromise;
  };

  // Returns today's slug, choosing (and recording) one if needed.
  function pickToday(cur) {
    const s = load();
    const d = todayStr();
    if (s.today && s.today.date === d && cur.notes[s.today.slug]) return s.today.slug;
    if (!cur.order.length) return null;
    let seen = new Set(s.seen);
    let next = cur.order.find((slug) => !seen.has(slug));
    if (!next) {
      s.cycle = (s.cycle || 1) + 1;
      seen = new Set();
      next = cur.order[0];
    }
    seen.add(next);
    s.seen = [...seen];
    s.today = { date: d, slug: next };
    save(s);
    return next;
  }

  function markSeen(slug, value = true) {
    const s = load();
    const seen = new Set(s.seen);
    value ? seen.add(slug) : seen.delete(slug);
    s.seen = [...seen];
    save(s);
  }

  function decorateSidebar() {
    const seen = new Set(load().seen);
    document.querySelectorAll(".md-nav__link[href]").forEach((a) => {
      const m = a.getAttribute("href").match(/notes\/([^/#]+)\/?$/);
      a.classList.toggle("dc-seen", !!(m && seen.has(m[1])));
    });
  }

  function renderReadToggle(el) {
    const slug = el.dataset.slug;
    const draw = () => {
      const isSeen = load().seen.includes(slug);
      el.innerHTML = "";
      const b = document.createElement("button");
      b.className = "md-button";
      b.textContent = isSeen ? "✓ Read — mark as unread" : "Mark as read";
      b.onclick = () => { markSeen(slug, !isSeen); draw(); decorateSidebar(); };
      el.appendChild(b);
    };
    draw();
  }

  async function renderHome(el) {
    const cur = await curriculum();
    const s = load();
    const total = cur.order.length;
    const seenCount = cur.order.filter((x) => s.seen.includes(x)).length;
    const pickedToday = s.today && s.today.date === todayStr() && cur.notes[s.today.slug];
    const root = siteRoot();
    const label = pickedToday
      ? `Today: <strong>${cur.notes[s.today.slug].title}</strong>`
      : "A new note is waiting for you.";
    el.innerHTML = `
      <p>${label}</p>
      <p><a class="md-button md-button--primary" href="${new URL("daily/", root)}">Open today's note →</a></p>
      <div class="dc-progress"><div style="width:${total ? (100 * seenCount) / total : 0}%"></div></div>
      <p class="dc-small">Cycle ${s.cycle || 1} · ${seenCount} of ${total} available notes read</p>`;
  }

  async function renderDaily(el) {
    const cur = await curriculum();
    const slug = pickToday(cur);
    if (!slug) { el.textContent = "No notes published yet."; return; }
    location.replace(new URL(cur.notes[slug].url, siteRoot()));
  }

  async function renderProgress(el) {
    const cur = await curriculum();
    const s = load();
    const seen = new Set(s.seen);
    const root = siteRoot();
    const rows = cur.order.map((slug, i) => {
      const n = cur.notes[slug];
      return `<tr><td>${i + 1}</td><td><a href="${new URL(n.url, root)}">${n.title}</a></td>
        <td>${n.track}</td><td>${seen.has(slug) ? "✓" : ""}</td></tr>`;
    });
    el.innerHTML = `
      <p>Cycle ${s.cycle || 1} · ${cur.order.filter((x) => seen.has(x)).length} / ${cur.order.length} read.
      Progress is stored in this browser only.</p>
      <table><thead><tr><th>#</th><th>Note</th><th>Track</th><th>Read</th></tr></thead>
      <tbody>${rows.join("")}</tbody></table>
      <p><button class="md-button" id="dc-reset">Reset progress</button></p>`;
    el.querySelector("#dc-reset").onclick = () => {
      if (confirm("Reset all reading progress in this browser?")) {
        save({ seen: [], today: null, cycle: 1 });
        renderProgress(el);
        decorateSidebar();
      }
    };
  }

  document$.subscribe(() => {
    const meta = document.querySelector(".note-meta[data-slug]");
    if (meta) markSeen(meta.dataset.slug);
    document.querySelectorAll(".read-toggle[data-slug]").forEach(renderReadToggle);
    const daily = document.getElementById("daily-redirect");
    if (daily) renderDaily(daily);
    const home = document.getElementById("today-card");
    if (home) renderHome(home);
    const prog = document.getElementById("progress-panel");
    if (prog) renderProgress(prog);
    decorateSidebar();
  });
})();
