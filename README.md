# PENDULUM.SYS — Auto-Balancer

An interactive inverted-pendulum game in a single HTML file. A real LQR + energy-pump
controller swings an n-link pendulum up from hanging, catches it upright, and balances
it on a cart — **your job is to knock it over**. It always gets back up.

**Play:** open [`index.html`](index.html) in any browser. No build, no dependencies.

## Controls

| Input | Action |
|---|---|
| Click/touch a bob and drag through it | Flick (longer drag = harder hit) |
| `←`/`→` or `A`/`D` | Shove the cart |
| `1`–`9`, `0` = 10, or the −/+ buttons | Number of links |
| `R` | Drop to hanging (it re-erects itself) |
| `C` | Manual crane hoist |
| `M` | Music + sound effects |

Score = total disturbance (N·s) the controller survives per balanced round.

## The physics is real

- Full nonlinear dynamics of an n-link pendulum on a cart (mass-matrix form,
  Gaussian elimination, RK4 at 240–3840 Hz fixed step depending on n).
- LQR gains computed offline with scipy (`solve_continuous_are` / `solve_discrete_are`)
  and validated against the nonlinear model — see the `*.py` design scripts.
- Swing-up uses energy-shaping control on the cart plus a soft joint compliance
  (spring–damper) active only during swing-up; joints also carry a small viscous friction.
- Flick impulses are applied physically: Δq̇ = M⁻¹Jᵀp.
- n = 1–4 swing themselves back up after any hit. n = 5 balances (recover with the crane).
  n ≥ 6 is a chaos sandbox: balancing 6+ frictionless links with one actuator is
  genuinely impossible — the gains blow past 10⁸ where float64 itself gives up.

## Sound

Theme music (132 BPM synthwave: kick/snare, 16th-note bass, arpeggios, lead, risers)
and all SFX are synthesized live with WebAudio — zero audio files.

---

Built with [Claude Code](https://claude.com/claude-code).
