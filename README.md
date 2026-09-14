# peiman-m.github.io

Source for my personal site. Three files, no build step, no dependencies.

- `index.html`: the page
- `style.css`: the stylesheet
- `background.js`: the animated background

To view it locally, open `index.html` in a browser. There is nothing to install and no
server to run; the only external request is the webfont.

## The background

Two points are picked just off the screen edge: left and right on a landscape screen, top
and bottom near the right side on a portrait one. The straight line between them is the
axis, and a Gaussian process models the perpendicular deviation from it:

```
P(t) = A + t (B - A) + f(t) n,    f ~ GP(0, k),   t in [0, 1]
```

The endpoints are observations pinned at `f = 0`. The path is then walked from start to
end: every few seconds a new observation appears, its value drawn from the *current*
posterior at that location, and the GP is refit. The kernel is an RBF; its lengthscale and
amplitude are drawn at random, along with the number and spacing of observations.

Nothing is drawn as a band. The page shows sixteen posterior samples as hairlines instead.
Writing the posterior covariance as `L Lᵀ`, each sample is the mean plus `L w` with
`w ~ N(0, I)`, so the threads pinch together at every observation and fan out wherever
nothing has been seen: uncertainty reads as spread. The RBF matters, since its samples are
smooth enough to stay clean as hairlines.

Two details do most of the work:

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
