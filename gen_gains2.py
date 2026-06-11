"""Free-joint (zero joint friction, zero joint torque) cart-pendulum controllers, n=1..10.

Cart force is the ONLY actuation. Per n:
  - LQR gains via scipy CARE
  - hold validation at a realistic browser control rate (ZOH)
  - max survivable tip kick (bisected) -> flick cap
  - swing-up capability: pure cart energy pumping + opportunistic catch
Emits a JS-ready JSON table.
"""
import json, sys
import numpy as np
from scipy.linalg import solve_continuous_are

G = 9.81

class Plant:
    def __init__(self, n):
        self.n = n
        self.Mc = 1.2
        self.m = [0.7/n]*n
        self.l = [1.35/n]*n
        self.cartFric = 0.675   # "normal" 0.45 x 1.5 per user request
        self.mu = [sum(self.m[j:]) for j in range(n)]
        self.Mtot = self.Mc + self.mu[0]
        self.Eup = sum(self.m[i]*G*2*sum(self.l[:i+1]) for i in range(n))

    def mass_matrix(self, q):
        n, l, mu = self.n, self.l, self.mu
        M = np.zeros((n+1, n+1))
        M[0, 0] = self.Mtot
        for j in range(n):
            M[0, j+1] = M[j+1, 0] = mu[j]*l[j]*np.cos(q[j+1])
            for k in range(n):
                M[j+1, k+1] = mu[max(j, k)]*l[j]*l[k]*np.cos(q[j+1]-q[k+1])
        return M

    def deriv(self, s, F):
        n = self.n
        q, qd = s[:n+1], s[n+1:]
        M = self.mass_matrix(q)
        b = np.zeros(n+1)
        b[0] = F - self.cartFric*qd[0]
        for j in range(n):
            th, w = q[j+1], qd[j+1]
            b[0] += self.mu[j]*self.l[j]*np.sin(th)*w*w
            b[j+1] = G*self.mu[j]*self.l[j]*np.sin(th)
            for k in range(n):
                b[j+1] -= self.mu[max(j, k)]*self.l[j]*self.l[k]*np.sin(th-q[k+1])*qd[k+1]**2
        return np.concatenate([qd, np.linalg.solve(M, b)])

    def lin(self):
        n = self.n
        M0 = self.mass_matrix(np.zeros(n+1))
        E = np.zeros((n+1, n+1))
        for j in range(n):
            E[j+1, j+1] = G*self.mu[j]*self.l[j]
        D = np.diag([self.cartFric] + [0.0]*n)
        c = np.zeros(n+1); c[0] = 1.0
        Mi = np.linalg.inv(M0)
        A = np.block([[np.zeros((n+1, n+1)), np.eye(n+1)], [Mi@E, -Mi@D]])
        B = np.concatenate([np.zeros(n+1), Mi@c])[:, None]
        return A, B

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

def lqr_u(p, K, s, Fmax):
    n = p.n
    sv = np.concatenate([[s[0]], wrap(s[1:n+1]), s[n+1:]])
    return float(np.clip(-K@sv, -Fmax, Fmax))

def rk4(p, s, u, h):
    k1 = p.deriv(s, u); k2 = p.deriv(s+h/2*k1, u)
    k3 = p.deriv(s+h/2*k2, u); k4 = p.deriv(s+h*k3, u)
    s = s + h/6*(k1+2*k2+2*k3+k4)
    n = p.n
    if s[0] > 2.0:  s[0] = 2.0;  s[n+1] = min(s[n+1], 0)*(-0.25)
    if s[0] < -2.0: s[0] = -2.0; s[n+1] = max(s[n+1], 0)*(-0.25)
    return s

def hold(p, K, Fmax, hz, s, seconds=6.0):
    """LQR-only; True if it stays up and settles."""
    h = 1/hz
    for i in range(int(seconds*hz)):
        u = lqr_u(p, K, s, Fmax)
        s = rk4(p, s, u, h)
        if not np.isfinite(s).all(): return False, s
        if np.abs(wrap(s[1:p.n+1])).max() > 1.0: return False, s
    ok = np.abs(wrap(s[1:p.n+1])).max() < 0.1
    return ok, s

def kick(p, s, mag):
    n = p.n
    M = p.mass_matrix(s[:n+1])
    Jx = np.zeros(n+1); Jx[0] = 1.0
    for j in range(n):
        Jx[j+1] = p.l[j]*np.cos(s[j+1])
    s = s.copy()
    s[n+1:] += np.linalg.solve(M, Jx*mag)
    return s

def max_kick(p, K, Fmax, hz):
    lo, hi = 0.0, 1.0
    for _ in range(7):
        mid = (lo+hi)/2
        s = kick(p, np.zeros(2*(p.n+1)), mid)
        ok, _ = hold(p, K, Fmax, hz, s, 6.0)
        if ok: lo = mid
        else: hi = mid
    return lo

def swingup(p, K, Fmax, Fsw, hz, seed, Tmax=120.0):
    """Pure cart pump, free joints, opportunistic catch with abort."""
    rng = np.random.default_rng(seed)
    n = p.n
    s = np.zeros(2*(n+1)); s[1:n+1] = np.pi + rng.uniform(-0.05, 0.05, n)
    h = 1/hz; mode = "swing"; stable = 0.0
    for i in range(int(Tmax*hz)):
        ths = wrap(s[1:n+1]); ws = s[n+2:]
        if mode == "swing":
            E = energy(p, s)
            a = -4.0*(p.Eup-E)*s[n+2]*np.cos(s[1]) - 1.2*s[0] - 1.5*s[n+1]
            u = float(np.clip(p.Mtot*a, -Fsw, Fsw))
            if (np.abs(ths) < 0.5).all() and (np.abs(ws) < 3.5).all() and abs(s[n+1]) < 2.0:
                mode = "lqr"
        if mode == "lqr":
            u = lqr_u(p, K, s, Fmax)
            if np.abs(ths).max() > 1.2: mode = "swing"; stable = 0.0
            elif np.abs(ths).max() < 0.1:
                stable += h
                if stable > 2.0: return i*h
        s = rk4(p, s, u, h)
        if not np.isfinite(s).all(): return None
    return None

if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1:]] or list(range(1, 11))
    HZ = {1:240, 2:240, 3:480, 4:960, 5:1920, 6:1920, 7:3840, 8:3840, 9:3840, 10:3840}
    out = {}
    for n in ns:
        p = Plant(n)
        A, B = p.lin()
        Q = np.diag([2.0] + [40.0*n]*n + [4.0] + [4.0]*n)
        R = 0.1
        P = solve_continuous_are(A, B, Q, np.array([[R]]))
        K = (B.T @ P / R).ravel()
        Fmax = min(30*n, 400)
        hz = HZ[n]
        ok, _ = hold(p, K, Fmax, hz, kick(p, np.zeros(2*(n+1)), 0.005), 6.0)
        mk = max_kick(p, K, Fmax, hz) if ok else 0.0
        # swing-up: only test where it has a prayer (n<=3), 2 seeds, 120 s
        ts = []
        if n <= 3 and ok:
            for seed in (0, 1):
                t = swingup(p, K, Fmax, 10+4*n, max(hz, 480), seed)
                ts.append(t)
        swing = len(ts) > 0 and all(t is not None for t in ts)
        print(f"n={n} hz={hz} hold={ok} maxKick={mk:.3f} swing={swing} ts={ts} Kmax={np.abs(K).max():.3g}",
              flush=True)
        out[n] = dict(K=[float(f"{k:.8g}") for k in K], m=p.m, l=p.l,
                      Fmax=Fmax, Fsw=10+4*n, hz=hz,
                      shove=8+3*n, maxKick=round(mk, 3), swing=bool(swing), hold=bool(ok))
    print(json.dumps(out))
