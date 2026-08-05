/* Background: sequential GP inference across the screen.
 *
 * Two points are chosen on the screen boundary. The straight line between them is taken
 * as the axis, and a Gaussian process models the perpendicular deviation from it:
 *
 *     P(t) = A + t (B - A) + f(t) n,      f ~ GP(0, k),   t in [0, 1]
 *
 * The endpoints are observations pinned at f = 0, so the curve genuinely starts and ends
 * on the boundary. From there the path is walked from start to end: every so often a new
 * observation is revealed, its value drawn from the current posterior at that location,
 * and the GP is refit. The shaded band is +/- 2 sd, so it collapses around each new
 * observation as it lands and stays wide wherever nothing has been seen yet.
 *
 * Nothing sits still. Writing the posterior covariance as L L^T, a perturbation
 *
 *     d = L w,     w ~ N(0, I)
 *
 * is a draw from the posterior, and w is rotated slowly between two fixed Gaussian
 * vectors so that it keeps its N(0, I) marginal while moving along a smooth path. Two
 * such fields are used, one nudging the mean and one breathing the band. Because L
 * carries the posterior's own correlation structure, the wobble is smooth along the path
 * and dies away at the observations by itself, so the dots stay pinned while everything
 * between them moves.
 *
 * Successive posteriors are eased into one another rather than swapped, so the band
 * visibly contracts instead of jumping. After the walk reaches the far end the path
 * fades, new boundary points and a new kernel are drawn, and it begins again.
 *
 * Plain canvas, no dependencies.
 */

(function () {
  'use strict';

  var canvas = document.getElementById('bg');
  if (!canvas || !canvas.getContext) return;
  var ctx = canvas.getContext('2d');

  var PATHS       = 1;
  var GRID        = 128;   // prediction points along the path
  var STEP_FRAMES = 320;   // frames between observations, roughly five seconds
  var EASE_FRAMES = 310;   // frames to ease from one posterior to the next, about five
  var BORN_FRAMES = 45;    // frames for a new observation dot to pop in
  var HOLD_FRAMES = 420;   // pause on the finished path, about seven seconds
  var FADE_FRAMES = 190;   // fade in / out, about three seconds each way
  var INTRO_FRAMES = 60;   // blank beat on first load before the curve fades in

  var MEAN_WIGGLE = 0.045; // mean drift, in units of posterior sd
  var BAND_WIGGLE = 0.065; // band breathing
  var SMOOTH_PASSES = 8;   // low-pass along the path, to kill grid-scale roughness

  var KERNELS = ['rbf', 'matern52'];

  var reduceMotion = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var accent = getComputedStyle(document.documentElement)
    .getPropertyValue('--link').trim() || '#008080';

  var w = 0, h = 0, dpr = 1;
  var paths = [];

  function rand(a, b) { return a + Math.random() * (b - a); }
  function randInt(a, b) { return a + Math.floor(Math.random() * (b - a + 1)); }

  function randn() {
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function smoothstep(x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    return x * x * (3 - 2 * x);
  }

  /* kernels (correlation only; amplitude applied separately) */

  function makeKernel(name) {
    var ell = rand(0.05, 0.20);
    if (name === 'rbf') {
      return function (a, b) { var d = (a - b) / ell; return Math.exp(-0.5 * d * d); };
    }
    // matern52: twice differentiable, so slightly less smooth than the RBF
    return function (a, b) {
      var d = Math.sqrt(5) * Math.abs(a - b) / ell;
      return (1 + d + d * d / 3) * Math.exp(-d);
    };
  }

  /* linear algebra */

  function cholesky(A, n) {
    var L = new Float64Array(n * n);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j <= i; j++) {
        var s = A[i * n + j];
        for (var k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        if (i === j) L[i * n + j] = Math.sqrt(Math.max(s, 1e-12));
        else         L[i * n + j] = s / (L[j * n + j] || 1e-9);
      }
    }
    return L;
  }

  function forwardSolve(L, b, n) {
    var y = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var s = b[i];
      for (var k = 0; k < i; k++) s -= L[i * n + k] * y[k];
      y[i] = s / (L[i * n + i] || 1e-9);
    }
    return y;
  }

  function backSolve(L, y, n) {
    var x = new Float64Array(n);
    for (var i = n - 1; i >= 0; i--) {
      var s = y[i];
      for (var k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
      x[i] = s / (L[i * n + i] || 1e-9);
    }
    return x;
  }

  // out = L v, for lower-triangular L (n x n).
  function lowerMul(L, v, n, out) {
    for (var i = 0; i < n; i++) {
      var acc = 0, row = i * n;
      for (var j = 0; j <= i; j++) acc += L[row + j] * v[j];
      out[i] = acc;
    }
    return out;
  }

  /* posterior on the grid */

  function posterior(p) {
    var n = p.obs.length, m = GRID, i, j, s;
    var k = p.k, amp2 = p.amp * p.amp, noise = amp2 * 1e-4;

    var Kcc = new Float64Array(n * n);
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        Kcc[i * n + j] = amp2 * k(p.obs[i].t, p.obs[j].t) + (i === j ? noise : 0);
      }
    }
    var Lc = cholesky(Kcc, n);

    var yv = new Float64Array(n);
    for (i = 0; i < n; i++) yv[i] = p.obs[i].y;
    var alpha = backSolve(Lc, forwardSolve(Lc, yv, n), n);

    // Cross-covariances, and V = Lc^{-1} Kgc^T.
    var mean = new Float64Array(m);
    var V = new Float64Array(n * m);
    var kx = new Float64Array(n);

    for (i = 0; i < m; i++) {
      var t = p.tg[i], mu = 0;
      for (j = 0; j < n; j++) kx[j] = amp2 * k(t, p.obs[j].t);
      for (j = 0; j < n; j++) mu += kx[j] * alpha[j];
      mean[i] = mu;

      var v = forwardSolve(Lc, kx, n);
      for (j = 0; j < n; j++) V[j * m + i] = v[j];
    }

    // cov = Kgg - V^T V
    var cov = new Float64Array(m * m);
    var sd = new Float64Array(m);
    for (i = 0; i < m; i++) {
      for (j = 0; j < m; j++) {
        var c = amp2 * k(p.tg[i], p.tg[j]);
        for (s = 0; s < n; s++) c -= V[s * m + i] * V[s * m + j];
        cov[i * m + j] = c;
      }
      cov[i * m + i] += amp2 * 1e-7;
      sd[i] = Math.sqrt(Math.max(cov[i * m + i], 0));
    }

    return { mean: mean, sd: sd, L: cholesky(cov, m) };
  }

  /* build a path */

  function boundaryPoint(edge) {
    if (edge === 0) return [rand(0, w), -8];          // top
    if (edge === 1) return [w + 8, rand(0, h)];       // right
    if (edge === 2) return [rand(0, w), h + 8];       // bottom
    return [-8, rand(0, h)];                          // left
  }

  // The whitened state is rotated smoothly between two fixed Gaussian vectors,
  //
  //     w(t) = cos(theta) a + sin(theta) b,     a, b ~ N(0, I),
  //
  // which keeps the N(0, I) marginal that makes L w a valid posterior draw, but moves
  // along a smooth path. An Ornstein-Uhlenbeck process would have the same marginal and
  // an arbitrarily long correlation time, yet its increments are white noise, so the
  // trajectory stays rough frame to frame however slowly it drifts. That roughness is
  // what reads as shimmer. Rotation has no high-frequency content at all.
  function makeField(m) {
    var a = new Float64Array(m), b = new Float64Array(m);
    for (var i = 0; i < m; i++) { a[i] = randn(); b[i] = randn(); }
    var f = {
      a: a, b: b,
      w: new Float64Array(m),
      theta: rand(0, Math.PI * 2),
      omega: (Math.PI * 2) / rand(2600, 6000),   // a full turn every 45-100 seconds
      cur: new Float64Array(m),
      old: new Float64Array(m)
    };
    syncField(f, m);
    return f;
  }

  function syncField(field, m) {
    var c = Math.cos(field.theta), s = Math.sin(field.theta);
    for (var i = 0; i < m; i++) field.w[i] = c * field.a[i] + s * field.b[i];
  }

  // How far a point can travel along (nx, ny) before leaving the viewport.
  function rayToEdge(x, y, nx, ny) {
    var d = Infinity;
    if (nx > 1e-9)       d = Math.min(d, (w - x) / nx);
    else if (nx < -1e-9) d = Math.min(d, (0 - x) / nx);
    if (ny > 1e-9)       d = Math.min(d, (h - y) / ny);
    else if (ny < -1e-9) d = Math.min(d, (0 - y) / ny);
    return Math.max(d, 0);
  }

  // The tightest perpendicular room the chord has, sampled where deviations are largest.
  function chordClearance(A, dx, dy, nx, ny) {
    var worst = Infinity;
    for (var s = 1; s <= 9; s++) {
      var t = 0.1 + 0.8 * (s / 10);
      var x = A[0] + dx * t, y = A[1] + dy * t;
      var room = Math.min(rayToEdge(x, y, nx, ny), rayToEdge(x, y, -nx, -ny));
      if (room < worst) worst = room;
    }
    return worst;
  }

  function makePath() {
    var diag = Math.sqrt(w * w + h * h);
    var best = null;

    // Try a few chords and keep the one with the most room to wander. Short chords that
    // clip a corner, or ones running close to an edge, score badly and get discarded.
    for (var attempt = 0; attempt < 14; attempt++) {
      var e1 = randInt(0, 3);
      var e2 = (e1 + randInt(1, 3)) % 4;              // a different edge
      var cA = boundaryPoint(e1), cB = boundaryPoint(e2);

      var cdx = cB[0] - cA[0], cdy = cB[1] - cA[1];
      var clen = Math.sqrt(cdx * cdx + cdy * cdy) || 1;
      var cnx = -cdy / clen, cny = cdx / clen;

      var clr = chordClearance(cA, cdx, cdy, cnx, cny);
      var score = clr * (clen >= 0.55 * diag ? 1 : 0.35);   // prefer real crossings

      if (!best || score > best.score) {
        best = { A: cA, dx: cdx, dy: cdy, len: clen, nx: cnx, ny: cny,
                 clr: clr, score: score };
      }
    }

    var A = best.A;
    var len = best.len;

    var tg = new Float64Array(GRID);
    for (var i = 0; i < GRID; i++) tg[i] = i / (GRID - 1);

    var p = {
      A: A,
      ux: best.dx / len, uy: best.dy / len,
      nx: best.nx, ny: best.ny,
      len: len,
      k: makeKernel(KERNELS[randInt(0, KERNELS.length - 1)]),
      // Prior sd is exactly `amp` (all kernels have k(t,t) = 1), so the band reaches
      // 2*amp. Budget a third of the available room per unit of sd and cap it, so a
      // cramped chord stays on screen and a central one still gets a full swing.
      amp: Math.max(
             Math.min(best.clr / 3.0, Math.min(w, h) * 0.13),
             Math.min(w, h) * 0.045
           ),
      tg: tg,
      obs: [{ t: 0, y: 0, born: 0 }, { t: 1, y: 0, born: 0 }],
      target: randInt(9, 14),
      frontier: rand(0, 0.10),
      timer: 0,
      nextStep: Math.round(STEP_FRAMES * rand(0.82, 1.35)),
      ease: 1,
      life: 0,
      dying: false,
      alpha: rand(0.085, 0.13),
      meanField: makeField(GRID),
      bandField: makeField(GRID)
    };

    var post = posterior(p);
    p.mean = post.mean; p.sd = post.sd; p.L = post.L;
    p.prevMean = post.mean.slice();
    p.prevSd = post.sd.slice();
    p.prevL = post.L;
    // What is actually on screen right now, so a refit can start from it.
    p.dispMean = post.mean.slice();
    p.dispSd = post.sd.slice();
    return p;
  }

  function nearest(t) {
    var i = Math.round(t * (GRID - 1));
    return i < 0 ? 0 : (i > GRID - 1 ? GRID - 1 : i);
  }

  function addObservation(p) {
    // Irregular spacing: mostly short hops that cluster, occasionally a long jump that
    // leaves a wide unobserved gap behind.
    var gap = (Math.random() < 0.55) ? rand(0.02, 0.06) : rand(0.09, 0.22);
    var t = p.frontier + gap;
    if (t > 0.95) t = 0.95;
    p.frontier = t;

    // Irregular in time too, so the reveals don't feel metronomic.
    p.nextStep = Math.round(STEP_FRAMES * rand(0.82, 1.35));

    var i = nearest(t);
    var y = p.mean[i] + p.sd[i] * randn();

    // Start the new transition from what is on screen, not from the previous target.
    // Otherwise a refit arriving mid-ease makes the curve jump.
    p.prevMean = p.dispMean.slice();
    p.prevSd = p.dispSd.slice();
    p.prevL = p.L;

    p.obs.push({ t: t, y: y, born: 0 });
    var post = posterior(p);
    p.mean = post.mean;
    p.sd = post.sd;
    p.L = post.L;
    p.ease = 0;
    p.target--;
  }

  /* simulation */

  // Measure the canvas box rather than window.innerHeight. The stylesheet pins the canvas
  // to the large viewport, so this holds still while a mobile browser's toolbars slide in
  // and out on scroll; innerHeight tracks them and jumps by ~60-100px instead.
  function measure() {
    var r = canvas.getBoundingClientRect();
    return [Math.round(r.width), Math.round(r.height)];
  }

  function resize() {
    var m = measure();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = m[0];
    h = m[1];
    canvas.width  = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function advance(field) {
    field.theta += field.omega;
    syncField(field, GRID);
  }

  function step() {
    for (var i = 0; i < paths.length; i++) {
      var p = paths[i];

      advance(p.meanField);
      advance(p.bandField);

      if (p.dying) {
        p.life -= 1 / FADE_FRAMES;
        if (p.life <= 0) paths[i] = makePath();
        continue;
      }

      if (p.life < 1) p.life = Math.min(1, p.life + 1 / FADE_FRAMES);
      if (p.ease < 1) p.ease = Math.min(1, p.ease + 1 / EASE_FRAMES);

      for (var j = 0; j < p.obs.length; j++) {
        if (p.obs[j].born < 1) p.obs[j].born = Math.min(1, p.obs[j].born + 1 / BORN_FRAMES);
      }

      p.timer++;
      if (p.target > 0 && p.frontier < 0.95) {
        if (p.timer >= p.nextStep) { p.timer = 0; addObservation(p); }
      } else if (p.timer >= HOLD_FRAMES) {
        p.dying = true;
      }
    }
  }

  /* drawing */

  function px(p, i, f) {
    var d = p.tg[i] * p.len;
    return [p.A[0] + p.ux * d + p.nx * f, p.A[1] + p.uy * d + p.ny * f];
  }

  var mean = new Float64Array(GRID);
  var sd   = new Float64Array(GRID);
  var dMean = new Float64Array(GRID);
  var dBand = new Float64Array(GRID);
  var tmp   = new Float64Array(GRID);

  // Binomial blur along the path. Rough kernels give L w plenty of grid-scale detail;
  // this keeps the slow shape and drops the rest. Near the observations the field is
  // already ~0, so blurring cannot unpin them.
  function lowPass(arr, passes) {
    for (var p = 0; p < passes; p++) {
      tmp[0] = 0.75 * arr[0] + 0.25 * arr[1];
      for (var i = 1; i < GRID - 1; i++) {
        tmp[i] = 0.25 * arr[i - 1] + 0.5 * arr[i] + 0.25 * arr[i + 1];
      }
      tmp[GRID - 1] = 0.75 * arr[GRID - 1] + 0.25 * arr[GRID - 2];
      arr.set(tmp);
    }
  }

  function render() {
    ctx.clearRect(0, 0, w, h);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = accent;
    ctx.fillStyle = accent;

    for (var q = 0; q < paths.length; q++) {
      var p = paths[q];
      var a = p.alpha * smoothstep(p.life);
      if (a <= 0.001) continue;

      var e = smoothstep(p.ease);
      var i, pt;

      // Wobble, eased across the refit along with the posterior itself.
      lowerMul(p.prevL, p.meanField.w, GRID, p.meanField.old);
      lowerMul(p.L,     p.meanField.w, GRID, p.meanField.cur);
      lowerMul(p.prevL, p.bandField.w, GRID, p.bandField.old);
      lowerMul(p.L,     p.bandField.w, GRID, p.bandField.cur);

      for (i = 0; i < GRID; i++) {
        mean[i] = p.prevMean[i] + (p.mean[i] - p.prevMean[i]) * e;
        sd[i]   = p.prevSd[i]   + (p.sd[i]   - p.prevSd[i])   * e;
        dMean[i] = p.meanField.old[i] + (p.meanField.cur[i] - p.meanField.old[i]) * e;
        dBand[i] = p.bandField.old[i] + (p.bandField.cur[i] - p.bandField.old[i]) * e;
      }

      // Remember the eased posterior, before the wobble is layered on.
      p.dispMean.set(mean);
      p.dispSd.set(sd);

      lowPass(dMean, SMOOTH_PASSES);
      lowPass(dBand, SMOOTH_PASSES);

      for (i = 0; i < GRID; i++) {
        mean[i] += dMean[i] * MEAN_WIGGLE;
        sd[i] = Math.max(sd[i] + dBand[i] * BAND_WIGGLE * 0.5, 0);
      }

      // +/- 2 sd band.
      ctx.globalAlpha = a * 0.26;
      ctx.beginPath();
      for (i = 0; i < GRID; i++) {
        pt = px(p, i, mean[i] + 2 * sd[i]);
        if (i === 0) ctx.moveTo(pt[0], pt[1]); else ctx.lineTo(pt[0], pt[1]);
      }
      for (i = GRID - 1; i >= 0; i--) {
        pt = px(p, i, mean[i] - 2 * sd[i]);
        ctx.lineTo(pt[0], pt[1]);
      }
      ctx.closePath();
      ctx.fill();

      // Band edges.
      ctx.globalAlpha = a * 0.7;
      ctx.lineWidth = 0.9;
      ctx.setLineDash([]);
      for (var s = -1; s <= 1; s += 2) {
        ctx.beginPath();
        for (i = 0; i < GRID; i++) {
          pt = px(p, i, mean[i] + s * 2 * sd[i]);
          if (i === 0) ctx.moveTo(pt[0], pt[1]); else ctx.lineTo(pt[0], pt[1]);
        }
        ctx.stroke();
      }

      // Posterior mean, dashed.
      ctx.globalAlpha = a * 1.25;
      ctx.lineWidth = 1.3;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      for (i = 0; i < GRID; i++) {
        pt = px(p, i, mean[i]);
        if (i === 0) ctx.moveTo(pt[0], pt[1]); else ctx.lineTo(pt[0], pt[1]);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Observations, popping in as they land.
      for (i = 0; i < p.obs.length; i++) {
        var o = p.obs[i];
        var b = smoothstep(o.born);
        var d = o.t * p.len;
        var ox = p.A[0] + p.ux * d + p.nx * o.y;
        var oy = p.A[1] + p.uy * d + p.ny * o.y;

        ctx.globalAlpha = a * 1.8 * b;
        ctx.beginPath();
        ctx.arc(ox, oy, 3.1 * (0.6 + 0.4 * b), 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // Every constant above is counted in frames, which only means what it says at 60Hz.
  // A 120Hz display (ProMotion, most gaming monitors) fires rAF twice as often and
  // would run the whole thing at double speed. So step() is driven on a fixed 1/60 s
  // tick and the frame loop just decides how many ticks are owed; render() still runs
  // once per repaint, so a fast display buys smoothness rather than speed.
  var TICK = 1000 / 60;
  var owed = 0;
  var prev = 0;

  function frame(now) {
    var dt = prev ? now - prev : TICK;
    prev = now;

    // rAF stops while the tab is hidden, so the first frame back reports a gap of
    // however long it was away. Advancing that literally would fast-forward through
    // whole cycles at once; resume where it left off instead.
    if (dt > 250) dt = TICK;

    owed += dt;
    // Cap the catch-up so a slow frame cannot spiral into ever more work per frame.
    var n = 0;
    while (owed >= TICK && n < 4) { step(); owed -= TICK; n++; }
    if (owed >= TICK) owed = 0;

    render();
    requestAnimationFrame(frame);
  }

  // Negative life is a delay: smoothstep clamps it to invisible until it climbs
  // past zero, then the normal fade-in takes over. Resize rebuilds skip the intro
  // so dragging a window edge doesn't blank the background.
  function build(intro) {
    paths = [];
    for (var i = 0; i < PATHS; i++) {
      var p = makePath();
      p.life = intro ? -INTRO_FRAMES / FADE_FRAMES : 1;
      paths.push(p);
    }
  }

  resize();
  build(true);

  // Mobile browsers fire resize as their toolbars retract on scroll. Rebuilding there
  // would discard a curve mid-inference and start a new one on a new chord, which is
  // what the scroll ends up looking like. A genuine layout change (a rotation, a dragged
  // window edge) moves the width or moves the height by a lot; browser chrome does
  // neither, so leave the curve alone and just keep the canvas the right size.
  var lastW = w, lastH = h;

  window.addEventListener('resize', function () {
    var m = measure();
    // Dragging a window to a display of a different density fires resize without
    // moving the box, and the backing store still has to be reallocated for it.
    var scale = Math.min(window.devicePixelRatio || 1, 2);
    if (m[0] === lastW && m[1] === lastH && scale === dpr) return;

    var reflow = m[0] !== lastW || Math.abs(m[1] - lastH) > 0.2 * lastH;
    lastW = m[0];
    lastH = m[1];

    resize();
    if (reflow) build();
    if (reduceMotion) render();
  });

  if (reduceMotion) {
    for (var i = 0; i < paths.length; i++) {
      var p = paths[i];
      while (p.target > 0 && p.frontier < 0.95) addObservation(p);
      p.ease = 1; p.life = 1;
      for (var j = 0; j < p.obs.length; j++) p.obs[j].born = 1;
      p.meanField.omega = 0; p.bandField.omega = 0;
    }
    render();
  } else {
    requestAnimationFrame(frame);
  }
})();
