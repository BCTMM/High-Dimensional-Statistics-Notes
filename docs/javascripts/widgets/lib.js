// Small numerical + plotting toolkit shared by the interactive widgets.
// Exposes window.DC = { randn, gamma, chi, eigSym, eigTridiag, solve, Plot, ... }.

(() => {
  // ---------- random numbers ----------
  let spare = null;
  function randn() {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u, v, r;
    do { u = 2 * Math.random() - 1; v = 2 * Math.random() - 1; r = u * u + v * v; } while (r >= 1 || r === 0);
    const f = Math.sqrt((-2 * Math.log(r)) / r);
    spare = v * f;
    return u * f;
  }

  // Gamma(shape k, scale 1), Marsaglia–Tsang.
  function gamma(k) {
    if (k < 1) return gamma(k + 1) * Math.pow(Math.random(), 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = randn(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = Math.random();
      if (u < 1 - 0.0331 * x ** 4) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  // Chi distribution with (possibly non-integer) k degrees of freedom.
  const chi = (k) => Math.sqrt(2 * gamma(k / 2));

  // ---------- linear algebra ----------
  // Eigenvalues of a symmetric tridiagonal matrix (diag d, off-diagonal e of
  // length n-1) by the implicit QL algorithm. Returns sorted ascending.
  function eigTridiag(dIn, eIn) {
    const n = dIn.length;
    const d = Float64Array.from(dIn);
    const e = new Float64Array(n);
    for (let i = 0; i < n - 1; i++) e[i] = eIn[i];
    const sign = (a, b) => (b >= 0 ? Math.abs(a) : -Math.abs(a));
    for (let l = 0; l < n; l++) {
      let iter = 0, m;
      do {
        for (m = l; m < n - 1; m++) {
          const dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
          if (Math.abs(e[m]) <= Number.EPSILON * dd) break;
        }
        if (m !== l) {
          if (iter++ === 100) throw new Error("eigTridiag: no convergence");
          let g = (d[l + 1] - d[l]) / (2 * e[l]);
          let r = Math.hypot(g, 1);
          g = d[m] - d[l] + e[l] / (g + sign(r, g));
          let s = 1, c = 1, p = 0, i;
          for (i = m - 1; i >= l; i--) {
            let f = s * e[i];
            const b = c * e[i];
            e[i + 1] = r = Math.hypot(f, g);
            if (r === 0) { d[i + 1] -= p; e[m] = 0; break; }
            s = f / r; c = g / r;
            g = d[i + 1] - p;
            r = (d[i] - g) * s + 2 * c * b;
            p = s * r;
            d[i + 1] = g + p;
            g = c * r - b;
          }
          if (r === 0 && i >= l) continue;
          d[l] -= p; e[l] = g; e[m] = 0;
        }
      } while (m !== l);
    }
    return Array.from(d).sort((a, b) => a - b);
  }

  // Eigenvalues of a small dense symmetric matrix (array of rows), cyclic Jacobi.
  function eigSym(A) {
    const n = A.length;
    const a = A.map((r) => Float64Array.from(r));
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] ** 2;
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          if (Math.abs(a[p][q]) < 1e-300) continue;
          const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) {
            const akp = a[k][p], akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
          }
          for (let k = 0; k < n; k++) {
            const apk = a[p][k], aqk = a[q][k];
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
          }
        }
      }
    }
    return a.map((r, i) => r[i]).sort((x, y) => x - y);
  }

  // Eigenvalues of a dense symmetric matrix of any moderate size: Householder
  // reduction to tridiagonal form, then implicit QL. O(n^3), fine for n ≲ 500.
  function eigSymFast(A) {
    const n = A.length;
    const a = A.map((r) => Float64Array.from(r));
    const v = new Float64Array(n), p = new Float64Array(n);
    for (let k = 0; k < n - 2; k++) {
      let norm = 0;
      for (let i = k + 1; i < n; i++) norm += a[i][k] * a[i][k];
      norm = Math.sqrt(norm);
      if (norm === 0) continue;
      const alpha = a[k + 1][k] > 0 ? -norm : norm;
      let vv = 0;
      for (let i = k + 1; i < n; i++) { v[i] = a[i][k]; }
      v[k + 1] -= alpha;
      for (let i = k + 1; i < n; i++) vv += v[i] * v[i];
      if (vv === 0) continue;
      // p = (2/vv) A v ;  q = p − (vᵀp / vv) v ;  A ← A − v qᵀ − q vᵀ
      let vp = 0;
      for (let i = k + 1; i < n; i++) {
        let s = 0;
        const ai = a[i];
        for (let j = k + 1; j < n; j++) s += ai[j] * v[j];
        p[i] = (2 * s) / vv;
        vp += v[i] * p[i];
      }
      const K = vp / vv;
      for (let i = k + 1; i < n; i++) p[i] -= K * v[i];
      for (let i = k + 1; i < n; i++) {
        const ai = a[i];
        for (let j = k + 1; j < n; j++) ai[j] -= v[i] * p[j] + p[i] * v[j];
      }
      a[k + 1][k] = alpha;
    }
    const d = Array.from({ length: n }, (_, i) => a[i][i]);
    const e = Array.from({ length: n - 1 }, (_, i) => a[i + 1][i]);
    return eigTridiag(d, e);
  }

  // Solve A x = b (A square, rows) by Gaussian elimination with partial pivoting.
  function solve(A, b) {
    const n = A.length;
    const M = A.map((r, i) => [...r, b[i]]);
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      [M[col], M[piv]] = [M[piv], M[col]];
      for (let r = col + 1; r < n; r++) {
        const f = M[r][col] / M[col][col];
        if (f !== 0) for (let k = col; k <= n; k++) M[r][k] -= f * M[col][k];
      }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) {
      let s = M[r][n];
      for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
      x[r] = s / M[r][r];
    }
    return x;
  }

  function histogram(xs, lo, hi, bins) {
    const counts = new Array(bins).fill(0);
    const w = (hi - lo) / bins;
    let n = 0;
    for (const x of xs) {
      const i = Math.floor((x - lo) / w);
      if (i >= 0 && i < bins) counts[i]++;
      n++;
    }
    // Normalized to a density over all samples (mass outside [lo, hi] is lost).
    return counts.map((c, i) => ({ x0: lo + i * w, x1: lo + (i + 1) * w, y: c / (n * w) }));
  }

  // ---------- plotting ----------
  const PALETTE = ["#4f6bed", "#f57c00", "#2e9d6a", "#c2399a"];

  class Plot {
    constructor(container, { height = 260, xlabel = "", ylabel = "", yticks = true } = {}) {
      this.canvas = document.createElement("canvas");
      container.appendChild(this.canvas);
      this.height = height;
      this.xlabel = xlabel;
      this.ylabel = ylabel;
      this.yticks = yticks;
      this.drawFn = null;
      const redraw = () => this.drawFn && this.drawFn();
      new ResizeObserver(redraw).observe(container);
      new MutationObserver(redraw).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
    }
    colors() {
      const cs = getComputedStyle(this.canvas);
      return {
        fg: cs.getPropertyValue("--md-default-fg-color").trim() || "#222",
        light: cs.getPropertyValue("--md-default-fg-color--lighter").trim() || "#aaa",
        lightest: cs.getPropertyValue("--md-default-fg-color--lightest").trim() || "#eee",
      };
    }
    // fn(ctx, api) draws using api.X(x), api.Y(y) for data -> pixel.
    draw(xr, yr, fn) {
      this.drawFn = () => this._draw(xr, yr, fn);
      this.drawFn();
    }
    _draw([x0, x1], [y0, y1], fn) {
      const c = this.canvas, dpr = window.devicePixelRatio || 1;
      const W = c.parentElement.clientWidth || 600, H = this.height;
      c.width = W * dpr; c.height = H * dpr;
      c.style.height = H + "px";
      const ctx = c.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const col = this.colors();
      const m = { l: this.yticks ? 44 : 10, r: 10, t: 10, b: 36 };
      const X = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
      const Y = (y) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
      ctx.font = "11px sans-serif";
      ctx.fillStyle = col.fg; ctx.strokeStyle = col.lightest; ctx.lineWidth = 1;
      const ticks = (a, b, n = 5) => {
        const step = Math.pow(10, Math.floor(Math.log10((b - a) / n)));
        const k = [1, 2, 5, 10].find((f) => (b - a) / (f * step) <= n) * step;
        const out = [];
        for (let t = Math.ceil(a / k) * k; t <= b + 1e-9 * k; t += k) out.push(+t.toPrecision(12));
        return out;
      };
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      for (const t of ticks(x0, x1)) {
        ctx.beginPath(); ctx.moveTo(X(t), m.t); ctx.lineTo(X(t), H - m.b); ctx.stroke();
        ctx.fillText(String(t), X(t), H - m.b + 4);
      }
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      for (const t of this.yticks ? ticks(y0, y1, 4) : []) {
        ctx.beginPath(); ctx.moveTo(m.l, Y(t)); ctx.lineTo(W - m.r, Y(t)); ctx.stroke();
        ctx.fillText(String(t), m.l - 4, Y(t));
      }
      ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText(this.xlabel, (m.l + W - m.r) / 2, H - 2);
      if (this.ylabel) {
        ctx.save(); ctx.translate(11, (m.t + H - m.b) / 2); ctx.rotate(-Math.PI / 2);
        ctx.textBaseline = "middle"; ctx.fillText(this.ylabel, 0, 0); ctx.restore();
      }
      ctx.save();
      ctx.beginPath(); ctx.rect(m.l, m.t, W - m.l - m.r, H - m.t - m.b); ctx.clip();
      fn(ctx, {
        X, Y, col, W, H,
        bars(hist, color = PALETTE[0], alpha = 0.45) {
          ctx.fillStyle = color; ctx.globalAlpha = alpha;
          for (const b of hist) ctx.fillRect(X(b.x0), Y(b.y), X(b.x1) - X(b.x0) - 0.5, Y(0) - Y(b.y));
          ctx.globalAlpha = 1;
        },
        line(xs, ys, color = PALETTE[1], width = 2, dash = []) {
          ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
          ctx.beginPath();
          xs.forEach((x, i) => (i ? ctx.lineTo(X(x), Y(ys[i])) : ctx.moveTo(X(x), Y(ys[i]))));
          ctx.stroke(); ctx.setLineDash([]);
        },
        curve(f, color = PALETTE[1], width = 2, dash = [], n = 300) {
          const xs = Array.from({ length: n + 1 }, (_, i) => x0 + ((x1 - x0) * i) / n);
          this.line(xs, xs.map(f), color, width, dash);
        },
        dots(xs, ys, color = PALETTE[0], r = 2.5, alpha = 0.6) {
          ctx.fillStyle = color; ctx.globalAlpha = alpha;
          xs.forEach((x, i) => { ctx.beginPath(); ctx.arc(X(x), Y(ys[i]), r, 0, 2 * Math.PI); ctx.fill(); });
          ctx.globalAlpha = 1;
        },
      });
      ctx.restore();
    }
  }

  // Legend / caption helper: [{color, label, dash}] -> HTML
  function legend(items) {
    return items
      .map(({ color, label, dash }) =>
        `<span style="display:inline-flex;align-items:center;gap:.3em;margin-right:1em">` +
        `<span style="display:inline-block;width:1.4em;border-top:${dash ? "2px dashed" : "3px solid"} ${color}"></span>${label}</span>`)
      .join("");
  }

  // Build a controls bar. spec: [{type:'range', id, label, min, max, step, value, fmt}, {type:'button', id, label}, {type:'select', id, label, options:[[value,text]]}]
  function controls(root, spec, onChange) {
    const bar = document.createElement("div");
    bar.className = "controls";
    const vals = {};
    for (const s of spec) {
      if (s.type === "range") {
        const lab = document.createElement("label");
        const out = document.createElement("span");
        const inp = Object.assign(document.createElement("input"), { type: "range", min: s.min, max: s.max, step: s.step, value: s.value });
        const fmt = s.fmt || ((v) => v);
        const upd = () => { vals[s.id] = +inp.value; out.textContent = fmt(+inp.value); };
        upd();
        inp.addEventListener("input", () => { upd(); onChange(s.id, vals); });
        lab.append(s.label + " ", inp, out);
        bar.appendChild(lab);
      } else if (s.type === "select") {
        const lab = document.createElement("label");
        const sel = document.createElement("select");
        for (const [v, t] of s.options) sel.appendChild(Object.assign(document.createElement("option"), { value: v, textContent: t }));
        sel.value = s.value;
        vals[s.id] = s.value;
        sel.addEventListener("change", () => { vals[s.id] = sel.value; onChange(s.id, vals); });
        lab.append(s.label + " ", sel);
        bar.appendChild(lab);
      } else if (s.type === "button") {
        const b = Object.assign(document.createElement("button"), { textContent: s.label, type: "button" });
        b.addEventListener("click", () => onChange(s.id, vals));
        bar.appendChild(b);
      }
    }
    root.appendChild(bar);
    return vals;
  }

  window.DC = { randn, gamma, chi, eigTridiag, eigSym, eigSymFast, solve, histogram, Plot, PALETTE, legend, controls, widgets: {} };
})();
