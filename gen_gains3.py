"""Joint-friction tuning + controller regeneration.

Joints get real viscous friction d (N*m per rad/s). Friction aligns the chain
naturally during swing-up (honest physics, no hidden torques), making catches
fast. Per n: find the smallest d giving quick reliable swing-up, design LQR
for the damped plant, measure basin.
"""
import json, sys
import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are, expm

G = 9.81

class Plant:
    def __init__(self, n, d):
        self.n = n; self.d = d
        self.Mc = 1.2
        self.m = [0.7/n]*n
        self.l = [1.35/n]*n
        self.cartFric = 0.675
        self.mu = [sum(self.m[j:]) for j in range(n)]
        self.Mtot = self.Mc + self.mu[0]
        self.Eup = sum(self.m[i]*G*2*sum(self.l[:i+1]) for i in range(n))

    def mass_matrix(self, q):
        n, l, mu = self.n, self.l, self.mu
        M = np.zeros((n+1, n+1)); M[0,0] = self.Mtot
        for j in range(n):
            M[0, j+1] = M[j+1, 0] = mu[j]*l[j]*np.cos(q[j+1])
            for k in range(n):
                M[j+1, k+1] = mu[max(j,k)]*l[j]*l[k]*np.cos(q[j+1]-q[k+1])
        return M

    def deriv(self, s, F):
        n = self.n
        q, qd = s[:n+1], s[n+1:]
        M = self.mass_matrix(q)
        b = np.zeros(n+1)
        b[0] = F - self.cartFric*qd[0]
        for j in range(n):
            th, w = q[j+1], qd[j+1]
            # viscous friction acts on RELATIVE joint velocity (physical joint friction)
            wrel_prev = w - (qd[j] if j > 0 else 0.0)
            b[0] += self.mu[j]*self.l[j]*np.sin(th)*w*w
            b[j+1] = G*self.mu[j]*self.l[j]*np.sin(th) - self.d*wrel_prev
            if j+1 < n:
                b[j+1] += self.d*(qd[j+2] - w)
            for k in range(n):
                b[j+1] -= self.mu[max(j,k)]*self.l[j]*self.l[k]*np.sin(th-q[k+1])*qd[k+1]**2
        return np.concatenate([qd, np.linalg.solve(M, b)])

    def lin(self):
        n = self.n
        M0 = self.mass_matrix(np.zeros(n+1))
        E = np.zeros((n+1, n+1))
        for j in range(n):
            E[j+1, j+1] = G*self.mu[j]*self.l[j]
        # damping matrix incl. relative-velocity joint friction
        D = np.zeros((n+1, n+1)); D[0,0] = self.cartFric
        for j in range(n):
            D[j+1, j+1] += self.d
            if j > 0: D[j+1, j] -= self.d
            if j+1 < n:
                D[j+1, j+1] += self.d
                D[j+1, j+2] -= self.d
        c = np.zeros(n+1); c[0] = 1.0
        Mi = np.linalg.inv(M0)
        A = np.block([[np.zeros((n+1, n+1)), np.eye(n+1)], [Mi@E, -Mi@D]])
        B = np.concatenate([np.zeros(n+1), Mi@c])[:, None]
        return A, B

def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi

def energy(p, s):
    n = p.n; q, qd = s[:n+1], s[n+1:]
    T = 0.5*p.Mc*qd[0]**2; V = 0.0
    vx, vy, hgt, hmax = qd[0], 0.0, 0.0, 0.0
    for i in range(n):
        vx += p.l[i]*np.cos(q[i+1])*qd[i+1]
        vy -= p.l[i]*np.sin(q[i+1])*qd[i+1]
        hgt += p.l[i]*np.cos(q[i+1]); hmax += p.l[i]
        T += 0.5*p.m[i]*(vx*vx + vy*vy)
        V += p.m[i]*G*(hgt + hmax)
    return T + V

def design(p, hz, qth, R, discrete):
    A, B = p.lin()
    if not discrete:
        Q = np.diag([2.0] + [qth]*p.n + [4.0] + [4.0]*p.n)
        P = solve_continuous_are(A, B, Q, np.array([[R]]))
        return (B.T @ P / R).ravel()
    h = 1/hz; nx = A.shape[0]
    M = np.zeros((nx+1, nx+1)); M[:nx,:nx] = A*h; M[:nx, nx:] = B*h
    E = expm(M); Ad, Bd = E[:nx,:nx], E[:nx,nx:]
    Q = np.diag([2.0] + [qth]*p.n + [4.0] + [4.0]*p.n)*h
    Rd = np.array([[R*h]])
    P = solve_discrete_are(Ad, Bd, Q, Rd)
    return (np.linalg.solve(Rd + Bd.T@P@Bd, Bd.T@P@Ad)).ravel()

def rk4(p, s, u, h):
    k1 = p.deriv(s, u); k2 = p.deriv(s+h/2*k1, u)
    k3 = p.deriv(s+h/2*k2, u); k4 = p.deriv(s+h*k3, u)
    s = s + h/6*(k1+2*k2+2*k3+k4)
    n = p.n
    if s[0] > 2.0:  s[0] = 2.0;  s[n+1] = min(s[n+1], 0)*(-0.25)
    if s[0] < -2.0: s[0] = -2.0; s[n+1] = max(s[n+1], 0)*(-0.25)
    return s

def lqr_u(p, K, s, Fmax):
    n = p.n
    sv = np.concatenate([[s[0]], wrap(s[1:n+1]), s[n+1:]])
    return float(np.clip(-K@sv, -Fmax, Fmax))

def swingup(p, K, Fmax, Fsw, hz, thC, wC, seed, Tmax=40.0):
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
            if (np.abs(ths) < thC).all() and (np.abs(ws) < wC).all() and abs(s[n+1]) < 2.0:
                mode = "lqr"
        if mode == "lqr":
            u = lqr_u(p, K, s, Fmax)
            if np.abs(ths).max() > 1.2: mode = "swing"; stable = 0.0
            elif np.abs(ths).max() < 0.1:
                stable += h
                if stable > 2.0: return round(i*h, 1)
        s = rk4(p, s, u, h)
        if not np.isfinite(s).all(): return None
    return None

def hold_kick(p, K, Fmax, hz, mag, seconds=5.0):
    n = p.n
    s = np.zeros(2*(n+1))
    M = p.mass_matrix(s[:n+1])
    Jx = np.zeros(n+1); Jx[0] = 1.0
    for j in range(n): Jx[j+1] = p.l[j]
    s[n+1:] += np.linalg.solve(M, Jx*mag)
    h = 1/hz
    for i in range(int(seconds*hz)):
        u = lqr_u(p, K, s, Fmax)
        s = rk4(p, s, u, h)
        if not np.isfinite(s).all(): return False
        if np.abs(wrap(s[1:n+1])).max() > 1.0: return False
    return np.abs(wrap(s[1:n+1])).max() < 0.15

if __name__ == "__main__":
    HZ   = {1:240, 2:240, 3:480, 4:960, 5:3840}
    FMAX = {1:30, 2:60, 3:90, 4:120, 5:200}
    TH   = {1:(0.5,3.5), 2:(0.5,3.5), 3:(0.4,3.0), 4:(0.35,2.5), 5:(0.3,2.0)}
    out = {}
    for n in ([int(x) for x in sys.argv[1:]] or [1,2,3,4,5]):
        hz, Fmax = HZ[n], FMAX[n]
        thC, wC = TH[n]
        best = None
        for d in (0.02, 0.05, 0.1, 0.2):
            p = Plant(n, d)
            K = design(p, hz, 40.0*n, 0.1, discrete=(n >= 5))
            if not hold_kick(p, K, Fmax, hz, 0.002):
                print(f"n={n} d={d}: hold FAIL", flush=True); continue
            ts = [swingup(p, K, Fmax, 10+4*n, max(hz,480), thC, wC, seed) for seed in (0,1)]
            ok = all(t is not None and t < 30 for t in ts)
            print(f"n={n} d={d}: swing {ts} {'OK' if ok else ''}", flush=True)
            if ok:
                lo, hi = 0.0, 1.2
                for _ in range(6):
                    mid = (lo+hi)/2
                    if hold_kick(p, K, Fmax, hz, mid, 5.0): lo = mid
                    else: hi = mid
                best = dict(d=d, K=[float(f"{k:.8g}") for k in K], hz=hz, Fmax=Fmax,
                            Fsw=10+4*n, thC=thC, wC=wC, maxKick=round(lo, 3),
                            ts=ts)
                break
        out[n] = best
        print(f"--> n={n}: {best and {k: best[k] for k in ('d','maxKick','ts')}}", flush=True)
    print(json.dumps(out))
