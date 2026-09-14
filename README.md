# peiman-m.github.io

Source for my personal site. Three files, no build step, no dependencies.

- `index.html`: the page
- `style.css`: the stylesheet
- `background.js`: the animated background

To view it locally, open `index.html` in a browser. There is nothing to install and no
server to run; the only external request is the webfont.

## The background

Two points are picked just off the screen edge: left and right on a landscape screen, top
and bottom on a portrait one, so that the straight line between them leans anywhere from
level to an 11° diagonal. That line is the axis, and two independent Gaussian processes
model the deviation from it, one across the screen and one into it:

```
P(t) = A + t (B - A) + f₁(t) n + f₂(t) z,    f₁, f₂ ~ GP(0, k),   t in [0, 1]
```

The endpoints are observations pinned at `f = 0`. The path is then walked from start to
end: every few seconds a new observation appears, a point in space whose two offsets are
drawn from the *current* posterior at that location, and the GP is refit. The kernel is an
RBF; its lengthscale and amplitude are drawn at random, along with the number and spacing of
observations.

Nothing is drawn as a band. The page shows sixteen posterior samples as hairlines instead.
Both offsets share the kernel and the observation locations, so they share one posterior
covariance `L Lᵀ`, and each sample is the pair of means plus `L w₁` and `L w₂` with
`w ~ N(0, I)`. The threads form a bundle, tied together at every observation and spread
wherever nothing has been seen: uncertainty reads as the width of the bundle. The RBF
matters, since its samples are smooth enough to stay clean as hairlines.

A still picture of a curve in space looks like a flat curve, so the depth has to be shown.
**The bundle rocks slowly about its own axis**, so near and far parts move against each
other while the pinned ends stay put. **It is drawn in perspective**, with nearer pieces of
thread a little darker and thicker. The observations are small shaded spheres, larger when
near, with the threads strung through them.

Two more details do most of the work:

**The threads follow each refit through a spring** rather than a timed ease. A critically
damped spring pulls the drawn mean and Cholesky factor toward the new posterior, so position
and speed both stay continuous: an observation landing mid-squeeze bends the motion instead
of stopping it or making it jump.

**The threads drift without shimmer.** Each `w` is *rotated* between two fixed Gaussian
vectors rather than driven as an Ornstein-Uhlenbeck process: same `N(0, I)` marginal, so
every frame is still a valid posterior draw, but OU has white-noise increments and so stays
rough frame to frame however slowly it drifts.

Amplitude is scaled to the perpendicular room the chord actually has, which keeps the
threads on screen. On wide screens the canvas fades to under half strength behind the text
column. The finished curve holds for about thirty seconds before the next one fades in.

`prefers-reduced-motion` resolves the whole walk at once and holds it still.
