// Interactive widgets. In a note:  <div class="widget" data-widget="NAME"></div>

(() => {
  const DC = window.DC;
  const W = DC.widgets;
  const [BLUE, ORANGE, GREEN, PINK] = DC.PALETTE;

  // log Gamma (Lanczos, g=7)
  const LG = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  function lgamma(x) {
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
    x -= 1;
    let a = LG[0];
    const t = x + 7.5;
    for (let i = 1; i < 9; i++) a += LG[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  const randSym = (n) => {
    const M = Array.from({ length: n }, () => Array.from({ length: n }, DC.randn));
    return M.map((r, i) => r.map((x, j) => (x + M[j][i]) / Math.SQRT2));
  };

  // ---------------------------------------------------------------------------
  // Weyl: eigenvalue paths of A + tE, with the Weyl band for one eigenvalue.
  W.weyl = (root) => {
    const n = 6;
    let A, E, paths, ts;
    const caption = document.createElement("div");
    const vals = DC.controls(root, [
      { type: "range", id: "k", label: "highlight λ", min: 1, max: n, step: 1, value: 3, fmt: (v) => `${v}` },
      { type: "button", id: "new", label: "New A and E" },
    ], (id) => { if (id === "new") sample(); render(); });
    const plot = new DC.Plot(root, { height: 280, xlabel: "t   (perturbation A + tE, with ‖E‖ = 1)", ylabel: "eigenvalues" });
    root.appendChild(caption);

    function sample() {
      // A: spread-out spectrum; E: random symmetric with operator norm 1.
      const Q = randSym(n);
      A = Q.map((r) => r.map((x) => x * 0.7));
      E = randSym(n);
      const nE = Math.max(...DC.eigSym(E).map(Math.abs));
      E = E.map((r) => r.map((x) => x / nE));
      ts = Array.from({ length: 121 }, (_, i) => (1.5 * i) / 120);
      paths = ts.map((t) => DC.eigSym(A.map((r, i) => r.map((x, j) => x + t * E[i][j]))));
    }
    function render() {
      const k = vals.k - 1;
      const all = paths.flat();
      const lo = Math.min(...all) - 0.5, hi = Math.max(...all) + 0.5;
      const lam0 = paths[0][k];
      plot.draw([0, 1.5], [lo, hi], (ctx, p) => {
        // Weyl band: |λ_k(A+tE) − λ_k(A)| ≤ t‖E‖
        ctx.fillStyle = ORANGE; ctx.globalAlpha = 0.12;
        ctx.beginPath();
        ctx.moveTo(p.X(0), p.Y(lam0));
        ctx.lineTo(p.X(1.5), p.Y(lam0 + 1.5));
        ctx.lineTo(p.X(1.5), p.Y(lam0 - 1.5));
        ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
        for (let j = 0; j < n; j++) {
          p.line(ts, paths.map((ev) => ev[j]), j === k ? ORANGE : BLUE, j === k ? 3 : 1.5);
        }
      });
      caption.className = "caption";
      caption.innerHTML = DC.legend([{ color: BLUE, label: "λ₁ ≤ … ≤ λ₆ of A + tE" }, { color: ORANGE, label: `λ${vals.k} and its Weyl band λ${vals.k}(A) ± t‖E‖` }]) +
        "<br>Sorted eigenvalue curves never cross; when two get close they veer apart (an <em>avoided crossing</em>). The highlighted curve never leaves its band.";
    }
    sample(); render();
  };

  // ---------------------------------------------------------------------------
  // High-dimensional geometry: angles between random vectors and norms.
  W["hd-geometry"] = (root) => {
    const pairs = 1500;
    const vals = DC.controls(root, [
      { type: "range", id: "logd", label: "dimension d", min: 0.31, max: 3, step: 0.01, value: 1, fmt: (v) => Math.round(10 ** v) },
      { type: "button", id: "re", label: "Resample" },
    ], () => render());
    const p1 = new DC.Plot(root, { height: 220, xlabel: "angle between two independent Gaussian vectors (degrees)", ylabel: "density" });
    const p2 = new DC.Plot(root, { height: 220, xlabel: "‖x‖ / √d", ylabel: "density" });
    const caption = document.createElement("div");
    caption.className = "caption";
    root.appendChild(caption);

    function render() {
      const d = Math.round(10 ** vals.logd);
      const angles = [], norms = [];
      for (let k = 0; k < pairs; k++) {
        let xy = 0, xx = 0, yy = 0;
        for (let i = 0; i < d; i++) { const x = DC.randn(), y = DC.randn(); xy += x * y; xx += x * x; yy += y * y; }
        angles.push((Math.acos(Math.max(-1, Math.min(1, xy / Math.sqrt(xx * yy)))) * 180) / Math.PI);
        norms.push(Math.sqrt(xx / d));
      }
      // angle density ∝ sin^{d−2} θ on [0, π]
      const logZ = 0.5 * Math.log(Math.PI) + lgamma((d - 1) / 2) - lgamma(d / 2);
      const angPdf = (deg) => {
        const th = (deg * Math.PI) / 180;
        if (d === 2) return 1 / 180;
        return Math.exp((d - 2) * Math.log(Math.sin(th)) - logZ) * (Math.PI / 180);
      };
      const hA = DC.histogram(angles, 0, 180, 60);
      const yA = Math.max(...hA.map((b) => b.y), angPdf(90)) * 1.15;
      p1.draw([0, 180], [0, yA], (ctx, p) => { p.bars(hA); p.curve(angPdf, ORANGE); });
      // r = ‖x‖/√d has density √d · χ_d(√d r)
      const normPdf = (r) => {
        if (r <= 0) return 0;
        const u = Math.sqrt(d) * r;
        return Math.sqrt(d) * Math.exp((d - 1) * Math.log(u) - (u * u) / 2 - (d / 2 - 1) * Math.LN2 - lgamma(d / 2));
      };
      const hN = DC.histogram(norms, 0, 2.5, 60);
      const yN = Math.max(...hN.map((b) => b.y), normPdf(1)) * 1.15;
      p2.draw([0, 2.5], [0, yN], (ctx, p) => { p.bars(hN, GREEN); p.curve(normPdf, ORANGE); });
      const sd = Math.sqrt(angles.reduce((s, a) => s + (a - 90) ** 2, 0) / pairs);
      caption.innerHTML = DC.legend([{ color: ORANGE, label: "exact density" }]) +
        `<br>d = ${d}: the angle has standard deviation ≈ ${sd.toFixed(1)}° (large-d approximation (180/π)/√d ≈ ${((180 / Math.PI) / Math.sqrt(d)).toFixed(1)}°). ` +
        "As d grows, random vectors are almost exactly orthogonal and almost exactly the same length.";
    }
    render();
  };

  // ---------------------------------------------------------------------------
  // β-ensembles: eigenvalue spacing distribution vs Poisson (Dumitriu–Edelman model).
  W["beta-spacing"] = (root) => {
    const N = 300, reps = 12;
    const vals = DC.controls(root, [
      { type: "select", id: "beta", label: "ensemble", value: "1",
        options: [["0", "β = 0: independent points (Poisson)"], ["1", "β = 1: GOE (real symmetric)"], ["2", "β = 2: GUE (complex Hermitian)"], ["4", "β = 4: GSE (quaternionic)"]] },
      { type: "button", id: "re", label: "Resample" },
    ], () => render());
    const rug = new DC.Plot(root, { height: 70, xlabel: "40 consecutive unfolded levels", yticks: false });
    const plot = new DC.Plot(root, { height: 250, xlabel: "spacing s (in units of the mean spacing)", ylabel: "density" });
    const caption = document.createElement("div");
    caption.className = "caption";
    root.appendChild(caption);

    const surmise = {
      0: (s) => Math.exp(-s),
      1: (s) => (Math.PI / 2) * s * Math.exp((-Math.PI * s * s) / 4),
      2: (s) => (32 / Math.PI ** 2) * s * s * Math.exp((-4 * s * s) / Math.PI),
      4: (s) => (2 ** 18 / (3 ** 6 * Math.PI ** 3)) * s ** 4 * Math.exp((-64 * s * s) / (9 * Math.PI)),
    };

    // Unfolded bulk eigenvalues: maps λ ↦ N·F(λ) with F the semicircle CDF, so the mean spacing is 1.
    function sampleUnfolded(beta) {
      if (beta === 0) {
        const xs = Array.from({ length: N }, () => Math.random() * N).sort((a, b) => a - b);
        return xs.slice(N / 4, (3 * N) / 4);
      }
      const d = Array.from({ length: N }, DC.randn);
      const e = Array.from({ length: N - 1 }, (_, k) => DC.chi(beta * (N - 1 - k)) / Math.SQRT2);
      const ev = DC.eigTridiag(d, e);
      const R = Math.sqrt(2 * beta * N);
      const F = (x) => { const u = Math.max(-1, Math.min(1, x / R)); return 0.5 + (u * Math.sqrt(1 - u * u) + Math.asin(u)) / Math.PI; };
      return ev.filter((x) => Math.abs(x) < R / 2).map((x) => N * F(x));
    }

    function render() {
      const beta = +vals.beta;
      const spacings = [];
      let first = null;
      for (let r = 0; r < reps; r++) {
        const u = sampleUnfolded(beta);
        if (!first) first = u;
        for (let i = 1; i < u.length; i++) spacings.push(u[i] - u[i - 1]);
      }
      const mid = Math.floor(first.length / 2) - 20;
      const pts = first.slice(mid, mid + 40);
      rug.draw([pts[0] - 0.5, pts[0] + 41], [0, 1], (ctx, p) => {
        ctx.strokeStyle = beta ? BLUE : GREEN; ctx.lineWidth = 2;
        for (const x of pts) { ctx.beginPath(); ctx.moveTo(p.X(x), p.Y(0.1)); ctx.lineTo(p.X(x), p.Y(0.9)); ctx.stroke(); }
      });
      const h = DC.histogram(spacings, 0, 4, 40);
      plot.draw([0, 4], [0, 1.25], (ctx, p) => {
        p.bars(h, beta ? BLUE : GREEN);
        p.curve(surmise[0], p.col.light, 1.5, [5, 4]);
        if (beta) p.curve(surmise[beta], ORANGE);
      });
      caption.innerHTML = DC.legend([
        { color: ORANGE, label: "Wigner surmise ∝ sᵝ e⁻ᶜˢ²" },
        { color: "gray", label: "Poisson e⁻ˢ", dash: true },
      ]) + `<br>${spacings.length} spacings from ${reps} matrices of size ${N}. ` +
        (beta ? `Small gaps are rare: the density vanishes like s${"⁰¹²³⁴"[beta]} near 0. That is level repulsion.`
              : "Independent points clump: small gaps are the most likely.");
    }
    render();
  };

  // ---------------------------------------------------------------------------
  // Markowitz: in-sample vs out-of-sample risk of the sample minimum-variance portfolio.
  W.markowitz = (root) => {
    const vals = DC.controls(root, [
      { type: "select", id: "N", label: "number of assets N", value: "50", options: [["20", "20"], ["50", "50"], ["100", "100"]] },
      { type: "button", id: "re", label: "Resample" },
    ], () => render());
    const plot = new DC.Plot(root, { height: 300, xlabel: "q = N / T   (assets per observation)", ylabel: "risk ÷ true optimal risk" });
    const caption = document.createElement("div");
    caption.className = "caption";
    root.appendChild(caption);

    // True covariance = I (the ratios below don't depend on Σ for Gaussian returns).
    function simulate(N, T) {
      const X = Array.from({ length: N }, () => Array.from({ length: T }, DC.randn));
      const Ehat = Array.from({ length: N }, () => new Array(N).fill(0));
      for (let i = 0; i < N; i++) for (let j = i; j < N; j++) {
        let s = 0; const xi = X[i], xj = X[j];
        for (let t = 0; t < T; t++) s += xi[t] * xj[t];
        Ehat[i][j] = Ehat[j][i] = s / T;
      }
      const y = DC.solve(Ehat, new Array(N).fill(1)); // Ê⁻¹ 1
      const s1 = y.reduce((a, b) => a + b, 0);         // 1ᵀ Ê⁻¹ 1
      const w = y.map((v) => v / s1);
      const inSample = 1 / s1;                          // wᵀ Ê w
      const outSample = w.reduce((a, v) => a + v * v, 0); // wᵀ Σ w with Σ = I
      const trueOpt = 1 / N;                            // 1 / (1ᵀ Σ⁻¹ 1)
      return [inSample / trueOpt, outSample / trueOpt];
    }

    function render() {
      const N = +vals.N;
      const qs = [], ins = [], outs = [];
      for (let q = 0.05; q < 0.92; q += 0.05) {
        for (let r = 0; r < 3; r++) {
          const T = Math.round(N / q);
          const [a, b] = simulate(N, T);
          qs.push(N / T); ins.push(a); outs.push(b);
        }
      }
      plot.draw([0, 1], [0, 5], (ctx, p) => {
        p.curve((q) => 1, p.col.light, 1.5, [5, 4]);
        p.curve((q) => 1 / (1 - q), ORANGE, 2, [], 400);
        p.curve((q) => 1 - q, BLUE);
        p.dots(qs, outs, ORANGE);
        p.dots(qs, ins, BLUE);
      });
      caption.innerHTML = DC.legend([
        { color: ORANGE, label: "out-of-sample (realized) risk: theory 1/(1−q)" },
        { color: BLUE, label: "in-sample (predicted) risk: theory 1−q" },
        { color: "gray", label: "true optimum", dash: true },
      ]) + "<br>Risk is measured as variance. Each dot is one simulated portfolio. The optimizer believes it beats the true optimum, but it actually does worse, and the gap blows up as q → 1.";
    }
    render();
  };

  // ---------------------------------------------------------------------------
  // Ridge: fit a noisy curve with 20 Gaussian bumps; slide lambda.
  W.ridge = (root) => {
    const n = 25, K = 20, width = 0.06, sigma = 0.3;
    const f = (x) => Math.sin(2 * Math.PI * x) + 0.5 * Math.cos(5 * x);
    const centers = Array.from({ length: K }, (_, k) => k / (K - 1));
    const feats = (x) => centers.map((c) => Math.exp(-((x - c) ** 2) / (2 * width * width)));
    let xs, ys;
    const vals = DC.controls(root, [
      { type: "range", id: "loglam", label: "log₁₀ λ", min: -6, max: 2, step: 0.1, value: -2, fmt: (v) => v.toFixed(1) },
      { type: "button", id: "re", label: "New data" },
    ], (id) => { if (id === "re") sample(); render(); });
    const plot = new DC.Plot(root, { height: 260, xlabel: "x" });
    const caption = document.createElement("div");
    caption.className = "caption";
    root.appendChild(caption);

    function sample() {
      xs = Array.from({ length: n }, () => Math.random());
      ys = xs.map((x) => f(x) + sigma * DC.randn());
    }
    function render() {
      const lam = 10 ** vals.loglam;
      const Phi = xs.map(feats);
      const G = centers.map((_, a) => centers.map((_, b) => Phi.reduce((s, r) => s + r[a] * r[b], 0) + (a === b ? lam : 0)));
      const rhs = centers.map((_, a) => Phi.reduce((s, r, i) => s + r[a] * ys[i], 0));
      const w = DC.solve(G, rhs);
      // df(λ) = tr(Φ (ΦᵀΦ + λI)⁻¹ Φᵀ) = Σ_i φ_iᵀ G⁻¹ φ_i
      let df = 0;
      for (const r of Phi) { const z = DC.solve(G, r); df += r.reduce((s, v, k) => s + v * z[k], 0); }
      const fit = (x) => feats(x).reduce((s, v, k) => s + v * w[k], 0);
      const wn = Math.sqrt(w.reduce((s, v) => s + v * v, 0));
      plot.draw([0, 1], [-2.2, 2.2], (ctx, p) => {
        p.curve(f, p.col.light, 1.5, [5, 4]);
        p.curve(fit, ORANGE, 2.5, [], 400);
        p.dots(xs, ys, BLUE, 3.5, 0.8);
      });
      caption.innerHTML = DC.legend([
        { color: BLUE, label: "data" }, { color: ORANGE, label: "ridge fit" }, { color: "gray", label: "true function", dash: true },
      ]) + `<br>λ = ${lam.toExponential(1)} · effective degrees of freedom df(λ) = ${df.toFixed(1)} (out of ${K}) · ‖w‖ = ${wn.toFixed(1)}. ` +
        "Small λ: the fit chases the noise with huge, cancelling weights. Large λ: everything is shrunk towards zero.";
    }
    sample(); render();
  };

  // ---------------------------------------------------------------------------
  // Split conformal: distribution of coverage over calibration draws (Beta law).
  W.conformal = (root) => {
    const trials = 400;
    const sd = (x) => 0.1 + 0.3 * x;              // noise level; model = true mean, score = |y − μ(x)|
    const erf = (x) => {                          // Abramowitz–Stegun 7.1.26
      const t = 1 / (1 + 0.3275911 * Math.abs(x));
      const y = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-x * x);
      return x >= 0 ? y : -y;
    };
    const xgrid = Array.from({ length: 200 }, (_, i) => (5 * (i + 0.5)) / 200);
    // Exact test coverage of the band ±q for x ~ Unif(0,5): average of P(|N(0, sd(x)²)| ≤ q).
    const coverage = (q) => xgrid.reduce((s, x) => s + erf(q / (sd(x) * Math.SQRT2)), 0) / xgrid.length;
    const vals = DC.controls(root, [
      { type: "range", id: "alpha", label: "α", min: 0.05, max: 0.3, step: 0.01, value: 0.1, fmt: (v) => v.toFixed(2) },
      { type: "range", id: "logn", label: "calibration size n", min: 1, max: 3, step: 0.05, value: 1.7, fmt: (v) => Math.round(10 ** v) },
      { type: "button", id: "re", label: "Resample" },
    ], () => render());
    const plot = new DC.Plot(root, { height: 260, xlabel: "test coverage of one conformal band (each trial = a fresh calibration set)", ylabel: "density" });
    const caption = document.createElement("div");
    caption.className = "caption";
    root.appendChild(caption);

    function render() {
      const alpha = vals.alpha, n = Math.round(10 ** vals.logn);
      const k = Math.ceil((n + 1) * (1 - alpha));
      const covs = [];
      for (let t = 0; t < trials; t++) {
        const scores = Array.from({ length: n }, () => { const x = 5 * Math.random(); return Math.abs(sd(x) * DC.randn()); }).sort((a, b) => a - b);
        covs.push(k > n ? 1 : coverage(scores[k - 1]));
      }
      const lo = Math.max(0, 1 - alpha - 0.3);
      const h = DC.histogram(covs, lo, 1, 40);
      const a = k, b = n - k + 1;                 // coverage ~ Beta(k, n − k + 1)
      const lB = lgamma(a) + lgamma(b) - lgamma(a + b);
      const beta = (c) => (c <= 0 || c >= 1 ? 0 : Math.exp((a - 1) * Math.log(c) + (b - 1) * Math.log(1 - c) - lB));
      const ymax = Math.max(...h.map((x) => x.y), beta(Math.min(0.999, (a - 1) / (a + b - 2)))) * 1.1;
      plot.draw([lo, 1], [0, ymax], (ctx, p) => {
        p.bars(h);
        if (k <= n) p.curve(beta, ORANGE, 2, [], 400);
        ctx.strokeStyle = GREEN; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(p.X(1 - alpha), p.Y(0)); ctx.lineTo(p.X(1 - alpha), p.Y(ymax)); ctx.stroke(); ctx.setLineDash([]);
      });
      const mean = covs.reduce((s, c) => s + c, 0) / trials;
      const below = covs.filter((c) => c < 1 - alpha - 0.02).length / trials;
      caption.innerHTML = DC.legend([{ color: ORANGE, label: `Beta(${a}, ${b}) theory` }, { color: GREEN, label: "target 1 − α", dash: true }]) +
        `<br>Average coverage ${mean.toFixed(3)} ≥ 1 − α = ${(1 - alpha).toFixed(2)} (the guarantee). ` +
        (k > n ? "n is too small for this α: the band is infinite." :
          `But for a single calibration set, coverage falls below ${(1 - alpha - 0.02).toFixed(2)} in ${(100 * below).toFixed(0)}% of trials. Larger n concentrates it.`);
    }
    render();
  };

  // ---------------------------------------------------------------------------
  // Wigner: eigenvalue histogram for different entry distributions vs the semicircle.
  W.wigner = (root) => {
    const samplers = {
      gauss: () => DC.randn(),
      rade: () => (Math.random() < 0.5 ? -1 : 1),
      sparse: (N) => { const p = 3 / N; return Math.random() < p ? (Math.random() < 0.5 ? -1 : 1) / Math.sqrt(p) : 0; },
      t: () => DC.randn() / Math.sqrt(DC.chi(2.5) ** 2 / 2.5) / Math.sqrt(5),
    };
    const vals = DC.controls(root, [
      { type: "select", id: "dist", label: "entries", value: "rade",
        options: [["gauss", "Gaussian"], ["rade", "Rademacher ±1"], ["sparse", "sparse: ~3 nonzeros per row"], ["t", "Student-t(2.5): heavy tails"]] },
      { type: "range", id: "N", label: "N", min: 20, max: 400, step: 10, value: 200 },
      { type: "button", id: "re", label: "Resample" },
    ], () => render());
    const plot = new DC.Plot(root, { height: 260, xlabel: "eigenvalue of X / √N", ylabel: "density" });
    const caption = document.createElement("div");
    caption.className = "caption";
    root.appendChild(caption);

    function render() {
      const N = vals.N, f = samplers[vals.dist];
      const A = Array.from({ length: N }, () => new Array(N).fill(0));
      for (let i = 0; i < N; i++) for (let j = i; j < N; j++) A[i][j] = A[j][i] = f(N) / Math.sqrt(N);
      const ev = DC.eigSymFast(A);
      const h = DC.histogram(ev, -3, 3, 60);
      const sc = (x) => (Math.abs(x) < 2 ? Math.sqrt(4 - x * x) / (2 * Math.PI) : 0);
      plot.draw([-3, 3], [0, Math.max(0.5, ...h.map((b) => b.y)) * 1.05], (ctx, p) => { p.bars(h); p.curve(sc, ORANGE, 2, [], 400); });
      const out = ev.filter((x) => Math.abs(x) > 2.2).length;
      caption.innerHTML = DC.legend([{ color: ORANGE, label: "semicircle" }]) +
        `<br>N = ${N}: λ_min = ${ev[0].toFixed(2)}, λ_max = ${ev[N - 1].toFixed(2)}, ${out} eigenvalue(s) beyond ±2.2. ` +
        ({ gauss: "The Gaussian case: semicircle with sharp edges at ±2.",
           rade: "Coin-flip entries give the same law: universality.",
           sparse: "With O(1) nonzeros per row the law is different (spiky, with tails). The semicircle needs the number of nonzeros per row → ∞.",
           t: "Finite variance but no 4th moment: big entries create outlier eigenvalues far beyond 2, and the bulk converges slowly." })[vals.dist];
    }
    render();
  };

  // ---------------------------------------------------------------------------
  document$.subscribe(() => {
    document.querySelectorAll(".widget[data-widget]").forEach((el) => {
      if (el.dataset.mounted) return;
      const f = W[el.dataset.widget];
      if (!f) { el.textContent = `Unknown widget: ${el.dataset.widget}`; return; }
      el.dataset.mounted = "1";
      try { f(el); } catch (e) { el.textContent = "Widget failed to load: " + e.message; console.error(e); }
    });
  });
})();
