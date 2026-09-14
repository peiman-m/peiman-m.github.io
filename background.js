/* Background: sequential GP inference across the screen, drawn as posterior samples in 3D.
 *
 * Two points are chosen just off the screen edge: left and right on a landscape screen,
 * top and bottom on a portrait one, so that the straight line between them leans anywhere
 * from level to a shallow diagonal. That line is taken as the axis, and two independent
 * Gaussian processes model the deviation from it, one along the in-screen normal n and one
 * along the depth axis z:
 *
 *     P(t) = A + t (B - A) + f1(t) n + f2(t) z,      f1, f2 ~ GP(0, k),   t in [0, 1]
 *
 * The endpoints are observations pinned at f = 0. From there the path is walked from
 * start to end: every so often a new observation is revealed, a point in space whose two
 * offsets are drawn from the current posterior at that location, and the GP is refit.
 *
 * Both offsets share the kernel and the observation locations, so they share the posterior
 * covariance too: one Cholesky factor serves both, and only the means differ. Writing that
 * covariance as L L^T, a sample is
 *
 *     (mean1 + L w1,  mean2 + L w2),     w1, w2 ~ N(0, I)
 *
 * and sixteen of them are stroked as hairlines. They form a bundle, tied together at every
 * observation and spread wherever nothing has been seen, so the uncertainty reads as the
 * width of the bundle. Each w is rotated slowly between two fixed Gaussian vectors, which
 * keeps its N(0, I) marginal, so every frame is still a valid posterior draw, while moving
 * along a smooth path.
 *
 * A still projection of a curve in space looks like a flat curve, so the depth is shown three
 * ways. The bundle rocks slowly about its own axis, so near and far parts move against each
 * other while the pinned ends stay put. It is projected in perspective. And nearer pieces of
 * thread are drawn darker and a little thicker, nearer observations as larger dots.
 *
 * The drawn posterior follows each refit through a critically damped spring rather than a
 * timed ease, so the bundle tightens smoothly, and an observation landing mid-squeeze only
 * redirects the motion instead of stopping it or making it jump. After the walk reaches the
 * far end the finished curve holds, fades, and a new one begins with fresh endpoints and
 * lengthscale.
 *
 * Plain canvas, no dependencies.
 */

(function () {
  'use strict';

  var canvas = document.getElementById('bg');
  if (!canvas || !canvas.getContext) return;
  var ctx = canvas.getContext('2d');

  var GRID         = 192;   // prediction points along the path
  var SAMPLES      = 16;    // posterior samples drawn as threads
  var STEP_FRAMES  = 330;   // frames between observations, about five and a half seconds
  var SETTLE_FRAMES = 450;  // frames for the threads to settle after a refit (95%), 7.5 s
  var BORN_FRAMES  = 150;   // frames for a new observation dot to fade in, about 2.5 seconds
  var HOLD_FRAMES  = 1800;  // pause on the finished curve, about thirty seconds
  var FADE_FRAMES  = 300;   // fade in / out, about five seconds each way
  var INTRO_FRAMES = 60;    // blank beat on first load before the threads fade in
  var SWING_FRAMES = 5400;  // one full rock of the bundle, there and back, ninety seconds

  var SWING  = 0.4;         // how far the bundle rocks either way about its axis, in radians
  var TILT_MIN = 0;         // each new curve leans between these many degrees, up or down
  var TILT_MAX = 20;
  var FOCAL  = 1.6;         // camera distance from the screen, in multiples of its long side
  var PIECE  = 8;           // grid points per stroked piece of thread, each shaded by depth

  var SMOOTH_PASSES = 3;    // low-pass along each thread, to kill grid-scale roughness

  // Spring rate per tick for a critically damped follow. From rest it covers 95% of the way
  // in SETTLE_FRAMES, since 1 - (1 + x) e^-x = 0.95 at x = 4.74.
  var OMEGA = 4.74 / SETTLE_FRAMES;

  var reduceMotion = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // The threads take their colour (--fg) and per-thread opacity (--curve) from the palette,
  // and the observation spheres are shaded between that ink and the page (--bg), with the
  // body --sphere of the way toward the ink. All of it changes with the theme, so it is read
  // again whenever that flips (see the observer near the bottom).
  var ink, inkRGB, paperRGB, opacity, sphereTone;
  function readPalette() {
    var root = getComputedStyle(document.documentElement);
    ink = root.getPropertyValue('--fg').trim() || '#171a21';
    inkRGB = parseHex(ink);
    paperRGB = parseHex(root.getPropertyValue('--bg').trim() || '#f3f4f7');
    opacity = parseFloat(root.getPropertyValue('--curve')) || 0.12;
    sphereTone = parseFloat(root.getPropertyValue('--sphere')) || 0.32;
  }

  // '#rrggbb' to [r, g, b], which is how the palette writes its colours.
  function parseHex(hex) {
    var m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
    return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [128, 128, 128];
  }

  function mix(a, b, t) {
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  }

  function rgb(c) {
    return 'rgb(' + Math.round(c[0]) + ',' + Math.round(c[1]) + ',' + Math.round(c[2]) + ')';
  }

  readPalette();

  var w = 0, h = 0, dpr = 1;
  var paths = [];

  function rand(a, b) { return a + Math.random() * (b - a); }
  function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }
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

  /* kernel (correlation only; amplitude applied separately) */

  // RBF only: its samples are infinitely smooth, which is what keeps hairlines clean. A
  // Matérn sample is rough at grid scale and reads as fuzz once drawn at full size.
  function makeKernel() {
    var ell = rand(0.04, 0.075);   // three to five bends across the path
    return function (a, b) { var d = (a - b) / ell; return Math.exp(-0.5 * d * d); };
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
    var n = p.obs.length, m = GRID, i, j, s, d;
    var k = p.k, amp2 = p.amp * p.amp, noise = amp2 * 1e-4;

    var Kcc = new Float64Array(n * n);
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        Kcc[i * n + j] = amp2 * k(p.obs[i].t, p.obs[j].t) + (i === j ? noise : 0);
      }
    }
    var Lc = cholesky(Kcc, n);

    // One set of weights per offset. Everything else in the fit is shared between them.
    var alpha = [], mean = [];
    for (d = 0; d < 2; d++) {
      var yv = new Float64Array(n);
      for (i = 0; i < n; i++) yv[i] = p.obs[i].y[d];
      alpha.push(backSolve(Lc, forwardSolve(Lc, yv, n), n));
      mean.push(new Float64Array(m));
    }

    // Cross-covariances, and V = Lc^{-1} Kgc^T.
    var V = new Float64Array(n * m);
    var kx = new Float64Array(n);

    for (i = 0; i < m; i++) {
      var t = p.tg[i];
      for (j = 0; j < n; j++) kx[j] = amp2 * k(t, p.obs[j].t);
      for (d = 0; d < 2; d++) {
        var mu = 0;
        for (j = 0; j < n; j++) mu += kx[j] * alpha[d][j];
        mean[d][i] = mu;
      }

      var v = forwardSolve(Lc, kx, n);
      for (j = 0; j < n; j++) V[j * m + i] = v[j];
    }

    // cov = Kgg - V^T V. An RBF covariance on a grid this fine is numerically close to
    // singular, and the threads are full-size draws from its Cholesky factor, so the
    // diagonal jitter is large enough (a sd floor of 0.3% of amp) to keep the factor clean.
    var cov = new Float64Array(m * m);
    var sd = new Float64Array(m);
    for (i = 0; i < m; i++) {
      for (j = 0; j < m; j++) {
        var c = amp2 * k(p.tg[i], p.tg[j]);
        for (s = 0; s < n; s++) c -= V[s * m + i] * V[s * m + j];
        cov[i * m + j] = c;
      }
      cov[i * m + i] += amp2 * 1e-5;
      sd[i] = Math.sqrt(Math.max(cov[i * m + i], 0));
    }

    return { mean: mean, sd: sd, L: cholesky(cov, m) };
  }

  /* build a path */

  // Each thread's whitened state, w1 and w2 end to end, is rotated smoothly between two
  // fixed Gaussian vectors,
  //
  //     w(t) = cos(theta) a + sin(theta) b,     a, b ~ N(0, I),
  //
  // which keeps the N(0, I) marginal that makes L w a valid posterior draw, but moves
  // along a smooth path. An Ornstein-Uhlenbeck process would have the same marginal and
  // an arbitrarily long correlation time, yet its increments are white noise, so the
  // trajectory stays rough frame to frame however slowly it drifts. That roughness is
  // what reads as shimmer. Rotation has no high-frequency content at all.
  function makeField(m) {
    var a = new Float64Array(2 * m), b = new Float64Array(2 * m);
    for (var i = 0; i < 2 * m; i++) { a[i] = randn(); b[i] = randn(); }
    var state = new Float64Array(2 * m);
    var f = {
      a: a, b: b,
      w: state,
      w1: state.subarray(0, m),
      w2: state.subarray(m),
      theta: rand(0, Math.PI * 2),
      omega: (Math.PI * 2) / rand(3600, 7200)    // a full turn every one to two minutes
    };
    syncField(f);
    return f;
  }

  function syncField(field) {
    var c = Math.cos(field.theta), s = Math.sin(field.theta);
    for (var i = 0; i < field.w.length; i++) field.w[i] = c * field.a[i] + s * field.b[i];
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
    var A, B, i;

    // Each curve leans anywhere from level to a shallow diagonal, one way or the other: across
    // a landscape screen, or down a portrait one. A level curve sits in the lower part of the
    // screen (or near the right edge); the steeper it leans, the closer to the middle it is
    // centred, so both ends stay on the page. The offset of each end is capped too, since on a
    // very wide or very tall screen those degrees would carry an end out past the corner.
    // The centre is then held where both ends land inside a band that leaves the threads room
    // to spread: on a phone, a steep lean would otherwise push one end against the right edge.
    var steep = rand(TILT_MIN, TILT_MAX);
    var lean = (Math.random() < 0.5 ? -1 : 1) * steep * Math.PI / 180;
    var f = steep / TILT_MAX, shift, c;
    if (w >= h) {
      shift = clamp((w / 2 + 12) * Math.tan(lean), -0.3 * h, 0.3 * h);
      c = clamp(h * rand(0.6 - 0.1 * f, 0.68 - 0.12 * f),
                0.2 * h + Math.abs(shift), 0.85 * h - Math.abs(shift));
      A = [-12, c - shift];
      B = [w + 12, c + shift];
    } else {
      shift = clamp((h / 2 + 12) * Math.tan(lean), -0.2 * w, 0.2 * w);
      c = clamp(w * rand(0.7 - 0.15 * f, 0.82 - 0.17 * f),
                0.35 * w + Math.abs(shift), 0.88 * w - Math.abs(shift));
      A = [c + shift, -12];
      B = [c - shift, h + 12];
    }

    var dx = B[0] - A[0], dy = B[1] - A[1];
    var len = Math.sqrt(dx * dx + dy * dy) || 1;
    var nx = -dy / len, ny = dx / len;
    var room = chordClearance(A, dx, dy, nx, ny);
    var short = Math.min(w, h);

    var tg = new Float64Array(GRID);
    for (i = 0; i < GRID; i++) tg[i] = i / (GRID - 1);

    var p = {
      A: A,
      ux: dx / len, uy: dy / len,
      nx: nx, ny: ny,
      len: len,
      k: makeKernel(),
      // Prior sd is exactly `amp` in every direction across the axis, so however the bundle
      // is turned, the widest threads reach about 2*amp from the axis on screen. Keep that
      // inside the room the chord has, and cap it at 13% of the screen's short side so the
      // swings are generous without taking over the page.
      amp: Math.max(Math.min(room / 2.5, short * 0.13), short * 0.05),
      tg: tg,
      obs: [{ t: 0, y: [0, 0], born: 1 }, { t: 1, y: [0, 0], born: 1 }],
      target: randInt(8, 12),
      frontier: rand(0, 0.08),
      timer: 0,
      nextStep: Math.round(STEP_FRAMES * rand(0.85, 1.3)),
      life: 0,
      dying: false,
      phase: rand(0, Math.PI * 2),   // where in its rock the bundle starts
      fields: []
    };
    for (i = 0; i < SAMPLES; i++) p.fields.push(makeField(GRID));

    var post = posterior(p);
    p.mean = post.mean; p.sd = post.sd; p.L = post.L;
    // What is drawn: the two means and a Cholesky factor that follow the posterior above
    // through a spring (see settle), with their velocities.
    p.drawMean = [post.mean[0].slice(), post.mean[1].slice()];
    p.drawL = post.L.slice();
    p.velMean = [new Float64Array(GRID), new Float64Array(GRID)];
    p.velL = new Float64Array(GRID * GRID);
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
    p.nextStep = Math.round(STEP_FRAMES * rand(0.85, 1.3));

    var i = nearest(t);
    var y = [p.mean[0][i] + p.sd[i] * randn(), p.mean[1][i] + p.sd[i] * randn()];

    // Only the posterior changes here. The drawn threads keep their position and velocity
    // and are pulled toward the new one from wherever they are (see settle).
    p.obs.push({ t: t, y: y, born: 0 });
    var post = posterior(p);
    p.mean = post.mean;
    p.sd = post.sd;
    p.L = post.L;
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
    syncField(field);
  }

  // One tick of a critically damped spring pulling x toward target. Position and velocity
  // stay continuous when the target changes, so a refit mid-squeeze bends the motion rather
  // than restarting it from a standstill. Blends of lower-triangular factors stay lower
  // triangular, so drawL is still a valid input to lowerMul.
  function follow(x, v, target) {
    var k = OMEGA * OMEGA, c = 2 * OMEGA;
    for (var i = 0; i < x.length; i++) {
      v[i] += k * (target[i] - x[i]) - c * v[i];
      x[i] += v[i];
    }
  }

  function settle(p) {
    follow(p.drawMean[0], p.velMean[0], p.mean[0]);
    follow(p.drawMean[1], p.velMean[1], p.mean[1]);
    follow(p.drawL, p.velL, p.L);
  }

  function step() {
    for (var i = 0; i < paths.length; i++) {
      var p = paths[i];

      for (var f = 0; f < p.fields.length; f++) advance(p.fields[f]);
      settle(p);
      p.phase += (Math.PI * 2) / SWING_FRAMES;

      if (p.dying) {
        p.life -= 1 / FADE_FRAMES;
        if (p.life <= 0) paths[i] = makePath();
        continue;
      }

      if (p.life < 1) p.life = Math.min(1, p.life + 1 / FADE_FRAMES);

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

  var dev1 = new Float64Array(GRID);
  var dev2 = new Float64Array(GRID);
  var tmp  = new Float64Array(GRID);
  var sx   = new Float64Array(GRID);   // projected thread, screen x
  var sy   = new Float64Array(GRID);   // projected thread, screen y
  var sz   = new Float64Array(GRID);   // depth before projection, positive away from the viewer

  // Binomial blur along the path. Near the observations the perturbation is already ~0,
  // so blurring cannot unpin the threads from the dots.
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

  // The camera looks at the middle of the canvas from FOCAL long sides away. The rock angle
  // of the path being drawn is kept as its cosine and sine. All are set in render().
  var camX = 0, camY = 0, focal = 1, rockC = 1, rockS = 0;
  var outX = 0, outY = 0, outZ = 0, outScale = 1;

  // A point `dist` along the axis with offsets f1 and f2: turn the offsets by the rock angle,
  // place the point in space, and project it. The result lands in outX, outY (screen),
  // outZ (depth), and outScale (how much perspective enlarges it).
  function place(p, dist, f1, f2) {
    var o = rockC * f1 - rockS * f2;   // along the in-screen normal
    var z = rockS * f1 + rockC * f2;   // into the screen
    var x = p.A[0] + p.ux * dist + p.nx * o;
    var y = p.A[1] + p.uy * dist + p.ny * o;
    outScale = focal / (focal + z);
    outX = camX + (x - camX) * outScale;
    outY = camY + (y - camY) * outScale;
    outZ = z;
  }

  // +1 for a point toward the viewer, -1 away, measured against the widest the threads reach.
  function nearness(p, z) {
    var v = -z / (2 * p.amp);
    return v < -1 ? -1 : (v > 1 ? 1 : v);
  }

  // A smooth stroke through the projected grid points, quadratic segments between midpoints,
  // cut into pieces of PIECE points so each can be shaded by its depth. Pieces meet at those
  // midpoints, where the curve is already smooth, and butt caps keep the joins from doubling
  // up into beads.
  function strokeThread(p, alpha) {
    var last = GRID - 1;
    for (var a = 1; a < last; a += PIECE) {
      var b = Math.min(a + PIECE, last), i, z = 0;
      for (i = a - 1; i <= b; i++) z += sz[i];
      var near = nearness(p, z / (b - a + 2));

      ctx.globalAlpha = Math.min(1, alpha * (1 + 0.45 * near));
      ctx.lineWidth = 0.7 * (1 + 0.25 * near);
      ctx.beginPath();
      if (a === 1) ctx.moveTo(sx[0], sy[0]);
      else ctx.moveTo((sx[a - 1] + sx[a]) / 2, (sy[a - 1] + sy[a]) / 2);
      for (i = a; i < b; i++) {
        ctx.quadraticCurveTo(sx[i], sy[i], (sx[i] + sx[i + 1]) / 2, (sy[i] + sy[i + 1]) / 2);
      }
      if (b === last) ctx.lineTo(sx[last], sy[last]);
      ctx.stroke();
    }
  }

  var WHITE = [255, 255, 255], BLACK = [0, 0, 0];

  // Observations, as small spheres fading in as they land. Each body sits between page and
  // ink, --sphere of the way along, and a radial gradient lights it from the upper left:
  // toward white at the highlight, toward black at the far rim, which reads as a ball in
  // either theme. They are opaque, so the threads look strung through them, and drawn far to
  // near, so a nearer sphere covers one behind it. The pinned ends sit just off screen.
  function drawDots(p, fade) {
    var dots = [], i;
    for (i = 0; i < p.obs.length; i++) {
      var o = p.obs[i];
      if (o.t <= 0 || o.t >= 1) continue;
      var b = smoothstep(o.born);
      if (b <= 0) continue;
      place(p, o.t * p.len, o.y[0], o.y[1]);
      dots.push({ x: outX, y: outY, z: outZ, scale: outScale, alpha: fade * b });
    }
    dots.sort(function (u, v) { return v.z - u.z; });

    for (i = 0; i < dots.length; i++) {
      var d = dots[i];
      var near = nearness(p, d.z);
      var r = 3.8 * d.scale * (1 + 0.25 * near);
      var body = mix(paperRGB, inkRGB, Math.min(1, sphereTone * (1 + 0.3 * near)));
      var shade = ctx.createRadialGradient(d.x - 0.35 * r, d.y - 0.4 * r, 0, d.x, d.y, r);
      shade.addColorStop(0, rgb(mix(body, WHITE, 0.35)));
      shade.addColorStop(0.5, rgb(body));
      shade.addColorStop(1, rgb(mix(body, BLACK, 0.25)));
      ctx.globalAlpha = d.alpha;
      ctx.fillStyle = shade;
      ctx.beginPath();
      ctx.arc(d.x, d.y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function render() {
    ctx.clearRect(0, 0, w, h);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'butt';
    ctx.strokeStyle = ink;
    camX = w / 2;
    camY = h / 2;
    focal = FOCAL * Math.max(w, h);

    for (var q = 0; q < paths.length; q++) {
      var p = paths[q];
      var fade = smoothstep(p.life);
      if (fade <= 0.001) continue;

      var rock = SWING * Math.sin(p.phase);
      rockC = Math.cos(rock);
      rockS = Math.sin(rock);

      // Each thread is the drawn means plus the drawn factor times that thread's whitened
      // states, one per offset.
      for (var s = 0; s < p.fields.length; s++) {
        var field = p.fields[s];
        lowerMul(p.drawL, field.w1, GRID, dev1);
        lowerMul(p.drawL, field.w2, GRID, dev2);
        lowPass(dev1, SMOOTH_PASSES);
        lowPass(dev2, SMOOTH_PASSES);
        for (var i = 0; i < GRID; i++) {
          place(p, p.tg[i] * p.len, p.drawMean[0][i] + dev1[i], p.drawMean[1][i] + dev2[i]);
          sx[i] = outX; sy[i] = outY; sz[i] = outZ;
        }
        strokeThread(p, Math.min(1, fade * opacity));
      }

      // Where every thread passes through one point they overlap into a knot; a sphere sits
      // over it.
      drawDots(p, fade);
    }
    ctx.globalAlpha = 1;
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
    paths = [makePath()];
    paths[0].life = intro ? -INTRO_FRAMES / FADE_FRAMES : 1;
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

  // The palette can change under a page that is already open: the toggle is clicked, or
  // the system setting flips while no choice is saved. Either way the theme script in
  // index.html sets a new data-theme on <html>. The animated loop picks the new colour up
  // on its next frame; the still one needs a repaint.
  function onTheme() {
    readPalette();
    if (reduceMotion) render();
  }
  if (window.MutationObserver) {
    new MutationObserver(onTheme).observe(document.documentElement,
      { attributes: true, attributeFilter: ['data-theme'] });
  }

  // Held still, the bundle stays at whatever rock angle it was built with; the perspective
  // and the depth shading still show which parts are near.
  if (reduceMotion) {
    var p = paths[0];
    while (p.target > 0 && p.frontier < 0.95) addObservation(p);
    p.drawMean[0].set(p.mean[0]); p.drawMean[1].set(p.mean[1]);
    p.drawL.set(p.L); p.life = 1;
    for (var j = 0; j < p.obs.length; j++) p.obs[j].born = 1;
    for (var f = 0; f < p.fields.length; f++) p.fields[f].omega = 0;
    render();
  } else {
    requestAnimationFrame(frame);
  }
})();
