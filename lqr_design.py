"""LQR gain design for n-link inverted pendulum on a cart.

Same model as index.html: point masses m[i] at the end of massless rods l[i],
angles measured from upright, generalized coords q = [x, th1..thn].
Linearized at upright:  M0 qdd = E q - D qd + c u
State s = [q, qd],  u = cart force.
Prints gain K and stress-tests the nonlinear closed loop with impulses.
"""
import numpy as np
from scipy.linalg import solve_continuous_are

G = 9.81

class Plant:
    def __init__(self, n):
        self.n = n
        if n == 2:
            self.Mc, self.m, self.l = 1.2, [0.35, 0.35], [0.55, 0.55]
        else:
            self.Mc, self.m, self.l = 1.2, [0.30, 0.25, 0.20], [0.45, 0.45, 0.45]
        self.cartFric, self.jointDamp = 0.45, 0.012
        self.mu = [sum(self.m[j:]) for j in range(n)]
        self.Mtot = self.Mc + self.mu[0]

    def mass_matrix(self, q):
        n, l, mu = self.n, self.l, self.mu
        M = np.zeros((n + 1, n + 1))
        M[0, 0] = self.Mtot
        for j in range(n):
            M[0, j + 1] = M[j + 1, 0] = mu[j] * l[j] * np.cos(q[j + 1])
            for k in range(n):
                M[j + 1, k + 1] = mu[max(j, k)] * l[j] * l[k] * np.cos(q[j + 1] - q[k + 1])
        return M

    def deriv(self, s, F):
        n = self.n
        q, qd = s[:n + 1], s[n + 1:]
        M = self.mass_matrix(q)
        b = np.zeros(n + 1)
        b[0] = F - self.cartFric * qd[0]
        for j in range(n):
            th, w = q[j + 1], qd[j + 1]
            b[0] += self.mu[j] * self.l[j] * np.sin(th) * w * w
            b[j + 1] = G * self.mu[j] * self.l[j] * np.sin(th) - self.jointDamp * w
            for k in range(n):
                b[j + 1] -= (self.mu[max(j, k)] * self.l[j] * self.l[k]
                             * np.sin(th - q[k + 1]) * qd[k + 1] ** 2)
        qdd = np.linalg.solve(M, b)
        return np.concatenate([qd, qdd])

    def lin(self):
        n = self.n
        M0 = self.mass_matrix(np.zeros(n + 1))
        E = np.zeros((n + 1, n + 1))
        for j in range(n):
            E[j + 1, j + 1] = G * self.mu[j] * self.l[j]
        D = np.diag([self.cartFric] + [self.jointDamp] * n)
        c = np.zeros(n + 1); c[0] = 1.0
        Mi = np.linalg.inv(M0)
        A = np.block([[np.zeros((n + 1, n + 1)), np.eye(n + 1)],
                      [Mi @ E, -Mi @ D]])
        B = np.concatenate([np.zeros(n + 1), Mi @ c])[:, None]
        return A, B


def design(n, qx, qth, qv, qw, R):
    p = Plant(n)
    A, B = p.lin()
    Q = np.diag([qx] + [qth] * n + [qv] + [qw] * n)
    P = solve_continuous_are(A, B, Q, np.array([[R]]))
    K = (B.T @ P / R).ravel()
    cl = np.linalg.eigvals(A - B @ K[None, :])
    return p, K, cl


def stress(p, K, Fmax, kick, bob, seconds=8.0):
    """Kick `bob` horizontally with impulse `kick` (N*s); return max |th| reached and survival."""
    n = p.n
    s = np.zeros(2 * (n + 1))
    # impulse -> generalized velocity change: M dqd = J^T pvec
    M = p.mass_matrix(s[:n + 1])
    Jx = np.zeros(n + 1); Jx[0] = 1.0
    for j in range(bob):
        Jx[j + 1] = p.l[j] * np.cos(s[j + 1])
    s[n + 1:] += np.linalg.solve(M, Jx * kick)
    h, peak = 1 / 480, 0.0
    for i in range(int(seconds / h)):
        u = float(np.clip(-K @ s, -Fmax, Fmax))
        k1 = p.deriv(s, u); k2 = p.deriv(s + h / 2 * k1, u)
        k3 = p.deriv(s + h / 2 * k2, u); k4 = p.deriv(s + h * k3, u)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        s[0] = np.clip(s[0], -2.0, 2.0)
        peak = max(peak, np.max(np.abs(s[1:n + 1])))
        if peak > 1.2:
            return peak, False
    settled = np.max(np.abs(s[1:n + 1])) < 0.05 and abs(s[0]) < 0.5
    return peak, settled


if __name__ == "__main__":
  for n, qth, Fmax in [(2, 80.0, 25.0), (3, 120.0, 50.0)]:
    p, K, cl = design(n, qx=2.0, qth=qth, qv=4.0, qw=4.0, R=0.05)
    print(f"\n=== n={n}  Fmax={Fmax} ===")
    print("K =", np.array2string(K, precision=4, separator=", "))
    print("closed-loop re(eig) max:", max(cl.real).round(3))
    for kick in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for bob in [1, n]:
            peak, ok = stress(p, K, Fmax, kick, bob)
            print(f"  kick {kick:4.1f} N*s @ bob {bob}: peak {peak:5.2f} rad  {'OK' if ok else 'TOPPLED'}")
