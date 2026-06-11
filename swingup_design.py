"""Swing-up controller prototyping: energy pumping + LQR catch.

Strategy:
  SWING-UP: cart acceleration a = ka*(Eup-E)*w1*cos(th1) - kx*x - kv*xdot,
            F = Mtot*a (saturated). Pumps total energy toward upright level
            while keeping the cart near center.
  CATCH:    when all |wrap(th_j)| < thCatch and |w_j| < wCatch and |xd| < vCatch,
            switch to LQR. If it diverges past 1.2 rad -> back to swing-up.
Measures time-to-stable-catch over random runs.
"""
import numpy as np
from lqr_design import Plant, design

G = 9.81

def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi

def energy(p, s):
    n = p.n
    q, qd = s[:n+1], s[n+1:]
    T = 0.5*p.Mc*qd[0]**2; V = 0.0
    vx, vy, hgt, hmax = qd[0], 0.0, 0.0, 0.0
    for i in range(n):
        vx += p.l[i]*np.cos(q[i+1])*qd[i+1]
        vy -= p.l[i]*np.sin(q[i+1])*qd[i+1]
        hgt += p.l[i]*np.cos(q[i+1]); hmax += p.l[i]
        T += 0.5*p.m[i]*(vx*vx + vy*vy)
        V += p.m[i]*G*(hgt + hmax)
    return T + V

def run(p, K, Fmax_lqr, Eup, ka, Fmax_sw, thCatch, wCatch, seed, Tmax=120.0):
    rng = np.random.default_rng(seed)
    n = p.n
    s = np.zeros(2*(n+1))
    s[1:n+1] = np.pi + rng.uniform(-0.05, 0.05, n)
    h = 1/240
    mode = "swing"
    stable_t = 0.0
    catches = 0
    for i in range(int(Tmax/h)):
        ths = wrap(s[1:n+1]); ws = s[n+1+1:]
        if mode == "swing":
            E = energy(p, s)
            a = -ka*(Eup - E)*s[n+2]*np.cos(s[1]) - 1.2*s[0] - 1.5*s[n+1]
            u = float(np.clip(p.Mtot*a, -Fmax_sw, Fmax_sw))
            if (np.abs(ths) < thCatch).all() and (np.abs(ws) < wCatch).all() and abs(s[n+1]) < 2.0:
                mode = "lqr"; catches += 1
        if mode == "lqr":
            sv = np.concatenate([[s[0]], ths, s[n+1:]])
            u = float(np.clip(-K @ sv, -Fmax_lqr, Fmax_lqr))
            if np.abs(ths).max() > 1.2:
                mode = "swing"; stable_t = 0.0
            elif np.abs(ths).max() < 0.1:
                stable_t += h
                if stable_t > 3.0:
                    return i*h, catches   # caught and held 3 s
        k1 = p.deriv(s, u); k2 = p.deriv(s + h/2*k1, u)
        k3 = p.deriv(s + h/2*k2, u); k4 = p.deriv(s + h*k3, u)
        s = s + h/6*(k1 + 2*k2 + 2*k3 + k4)
        if s[0] > 2.0:  s[0] = 2.0;  s[n+1] = min(s[n+1], 0)*(-0.25)
        if s[0] < -2.0: s[0] = -2.0; s[n+1] = max(s[n+1], 0)*(-0.25)
    return None, catches

if __name__ == "__main__":
    import sys, itertools
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    if n == 2:
        p, K, _ = design(2, qx=2.0, qth=80.0, qv=4.0, qw=4.0, R=0.05)
        Fl = 25.0
    else:
        p, K, _ = design(3, qx=2.0, qth=120.0, qv=4.0, qw=4.0, R=0.05)
        Fl = 80.0
    Eup = sum(p.m[i]*G*2*sum(p.l[:i+1]) for i in range(n))
    for ka, Fsw, thC, wC in itertools.product(
            [1.5, 4.0], [12.0, 20.0], [0.35, 0.55], [2.5, 4.5]):
        times = []
        for seed in range(6):
            t, c = run(p, K, Fl, Eup, ka, Fsw, thC, wC, seed)
            times.append(t)
        ok = [t for t in times if t is not None]
        print(f"ka={ka:4} Fsw={Fsw:4} thC={thC} wC={wC}: "
              f"caught {len(ok)}/6, t = {[round(t,1) for t in ok]}", flush=True)
