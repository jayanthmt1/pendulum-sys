"""Generate validated controller gain tables for n = 1..5 link cart-pendulums.

Per n: parametric plant (total reach 1.35 m, total bob mass 0.7 kg, cart 1.2 kg),
LQR via scipy CARE, swing-up = energy pump + alignment assist.
Validation: (a) swing-up from hanging -> stable catch,
            (b) tip impulse at full game flick strength (spins like hell) -> re-catch.
Outputs a JS-ready JSON table.
"""
import json
import numpy as np
from scipy.linalg import solve_continuous_are

G = 9.81

class Plant:
    def __init__(self, n):
        self.n = n
        self.Mc = 1.2
        self.m = [0.7/n]*n
        self.l = [1.35/n]*n
        self.cartFric, self.jointDamp = 0.45, 0.012
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

    def deriv(self, s, F, tau):
        n = self.n
        q, qd = s[:n+1], s[n+1:]
        M = self.mass_matrix(q)
        b = np.zeros(n+1)
        b[0] = F - self.cartFric*qd[0]
        for j in range(n):
            th, w = q[j+1], qd[j+1]
            b[0] += self.mu[j]*self.l[j]*np.sin(th)*w*w
            b[j+1] = G*self.mu[j]*self.l[j]*np.sin(th) - self.jointDamp*w + tau[j]
            for k in range(n):
                b[j+1] -= self.mu[max(j, k)]*self.l[j]*self.l[k]*np.sin(th-q[k+1])*qd[k+1]**2
        return np.concatenate([qd, np.linalg.solve(M, b)])

    def lin(self):
        n = self.n
        M0 = self.mass_matrix(np.zeros(n+1))
        E = np.zeros((n+1, n+1))
        for j in range(n):
            E[j+1, j+1] = G*self.mu[j]*self.l[j]
        D = np.diag([self.cartFric] + [self.jointDamp]*n)
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
    vx, vy, hgt = qd[0], 0.0, 0.0
    hmax = 0.0
    for i in range(n):
        vx += p.l[i]*np.cos(q[i+1])*qd[i+1]
        vy -= p.l[i]*np.sin(q[i+1])*qd[i+1]
        hgt += p.l[i]*np.cos(q[i+1]); hmax += p.l[i]
        T += 0.5*p.m[i]*(vx*vx + vy*vy)
        V += p.m[i]*G*(hgt + hmax)
    return T + V

def simulate(p, K, cfg, s, seconds, mode="swing"):
    """Run the full state machine; return (mode, s, caught_stable_time_total)."""
    h = 1/240
    stable = 0.0
    for i in range(int(seconds/h)):
        n = p.n
        ths = wrap(s[1:n+1]); ws = s[n+2:]
        tau = np.zeros(n)
        if mode == "swing":
            E = energy(p, s)
            a = -cfg["ka"]*(p.Eup - E)*s[n+2]*np.cos(s[1]) - 1.2*s[0] - 1.5*s[n+1]
            u = float(np.clip(p.Mtot*a, -cfg["Fsw"], cfg["Fsw"]))
            for j in range(1, n):
                d = wrap(s[j+1]-s[j]); dw = s[n+2+j]-s[n+1+j]
                tau[j] -= cfg["kal"]*d + cfg["cal"]*dw
                tau[j-1] += cfg["kal"]*d + cfg["cal"]*dw
            if (np.abs(ths) < 0.4).all() and (np.abs(ws) < 3.0).all() and abs(s[n+1]) < 2.0:
                mode = "lqr"
        if mode == "lqr":
            sv = np.concatenate([[s[0]], ths, s[n+1:]])
            u = float(np.clip(-np.array(K)@sv, -cfg["Fmax"], cfg["Fmax"]))
            if np.abs(ths).max() > 1.0:
                mode = "swing"; stable = 0.0
            elif np.abs(ths).max() < 0.1:
                stable += h
                if stable > 3.0:
                    return mode, s, i*h
        k1 = p.deriv(s, u, tau); k2 = p.deriv(s+h/2*k1, u, tau)
        k3 = p.deriv(s+h/2*k2, u, tau); k4 = p.deriv(s+h*k3, u, tau)
        s = s + h/6*(k1+2*k2+2*k3+k4)
        if s[0] > 2.0:  s[0] = 2.0;  s[n+1] = min(s[n+1], 0)*(-0.25)
        if s[0] < -2.0: s[0] = -2.0; s[n+1] = max(s[n+1], 0)*(-0.25)
    return mode, s, None

def kick(p, s, bob, mag):
    n = p.n
    M = p.mass_matrix(s[:n+1])
    Jx = np.zeros(n+1); Jx[0] = 1.0
    for j in range(bob):
        Jx[j+1] = p.l[j]*np.cos(s[j+1])
    s = s.copy()
    s[n+1:] += np.linalg.solve(M, Jx*mag)
    return s

table = {}
for n in range(1, 6):
    p = Plant(n)
    A, B = p.lin()
    Q = np.diag([2.0] + [40.0*n]*n + [4.0] + [4.0]*n)
    R = 0.05
    P = solve_continuous_are(A, B, Q, np.array([[R]]))
    K = (B.T @ P / R).ravel()
    Fmax = [0, 30, 60, 130, 200, 300][n]
    cfg = dict(ka=4.0, Fsw=10 + 4*n, kal=3.0*max(1, n-1), cal=0.3*max(1, n-1),
               Fmax=Fmax)
    flick = round(2.5/n, 2)

    # (a) swing-up from hanging
    s0 = np.zeros(2*(n+1)); s0[1:n+1] = np.pi + 0.03
    mode, s1, t_up = simulate(p, K, cfg, s0, 45.0)
    # (b) full-strength tip kick from balance -> wild spin -> must re-catch
    t_rec = None
    if t_up is not None:
        s2 = kick(p, s1, n, flick)
        _, _, t_rec = simulate(p, K, cfg, s2, 45.0, mode="lqr")
    ok = t_up is not None and t_rec is not None
    print(f"n={n}: swing-up {t_up and round(t_up,1)} s, "
          f"spin-recovery {t_rec and round(t_rec,1)} s  {'OK' if ok else 'FAIL'}", flush=True)
    table[n] = dict(K=[round(float(k), 6) for k in K],
                    m=p.m, l=p.l,
                    Fmax=cfg["Fmax"], Fsw=cfg["Fsw"],
                    ka=cfg["ka"], kal=cfg["kal"], cal=cfg["cal"],
                    shove=8 + 3*n, flickCap=flick, ok=ok)

print()
print(json.dumps(table))
