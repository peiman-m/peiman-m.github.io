# peiman-m.github.io

Source for my personal site. Three files, no build step, no dependencies.

- `index.html`: the page
- `style.css`: the stylesheet
- `background.js`: the animated background

To view it locally, open `index.html` in a browser. There is nothing to install and no
server to run; the only external request is the webfont.

## The background

Two points are picked on the screen boundary. The straight line between them is taken as
the axis, and a Gaussian process models the perpendicular deviation from it:

```
P(t) = A + t (B - A) + f(t) n,    f ~ GP(0, k),   t in [0, 1]
```

The endpoints are observations pinned at `f = 0`, so the curve really does start and end
on the boundary. The path is then walked from start to end: every few seconds a new
observation appears, its value drawn from the *current* posterior at that location, and
the GP is refit. The shaded band is ±2 standard deviations, so it collapses around each
new point and stays wide wherever nothing has been seen yet. Kernels are drawn from RBF
and Matérn-5/2, along with lengthscale, amplitude, and the number and spacing of
observations.

Two details do most of the work:

**Successive posteriors are eased into one another** rather than swapped, and each refit
starts from the curve currently on screen, so an observation landing mid-transition
cannot make the curve jump.

**The wobble comes from the posterior itself.** Writing the posterior covariance as
`L Lᵀ`, the perturbation `d = L w` with `w ~ N(0, I)` is a draw from the posterior, so it
is smooth along the path and vanishes at the observations by construction: the dots stay
pinned while everything between them drifts. `w` is *rotated* between two fixed Gaussian
vectors rather than driven as an Ornstein-Uhlenbeck process: same `N(0, I)` marginal, but
OU has white-noise increments and so stays rough frame to frame however slowly it drifts,
which reads as shimmer.

Amplitude is scaled to the perpendicular clearance the chord actually has rather than to a
fixed fraction of the viewport, which is what keeps curves from running off screen.

`prefers-reduced-motion` resolves the whole walk at once and holds it still.
