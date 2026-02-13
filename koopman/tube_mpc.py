import numpy as np
import cvxpy as cp
from .mpc_qp import LinearMPC, compute_LQR_gain

def compute_inf_norm_gain(A, B):
    """
    Compute K such that |A + BK|_inf is minimized.
    Formulated as LP.
    """
    nz = A.shape[0]
    nu = B.shape[1] if len(B.shape) > 1 else 1

    # Reshape B if needed
    if len(B.shape) == 1:
        B = B.reshape(-1, 1)

    K = cp.Variable((nu, nz))
    gamma = cp.Variable(nonneg=True)
    M = cp.Variable((nz, nz), nonneg=True)

    constraints = []
    # M >= |A + BK|
    term = A + B @ K
    constraints.append(M >= term)
    constraints.append(M >= -term)

    # Row sum constraint
    constraints.append(cp.sum(M, axis=1) <= gamma)

    # Regularization
    loss = gamma + 0.001 * cp.sum(cp.abs(K))

    prob = cp.Problem(cp.Minimize(loss), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        # Return -K because LQR uses u = -K x, here we solved A + BK
        # So K_fb = -K
        return -K.value, gamma.value
    else:
        return None, None

class TubeMPC:
    def __init__(self, A, B, Q, R, N, x_min, x_max, u_min, u_max, v_min, v_max, Cx, Cu,
                 w_bar_x, w_bar_u, w_bar_inf=None):
        """
        w_bar_x: bound on disturbance for x
        w_bar_u: bound on disturbance for u
        w_bar_inf: (optional) bound on disturbance |w|_inf for full state
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = N
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        self.v_min = v_min
        self.v_max = v_max
        self.Cx = Cx
        self.Cu = Cu
        self.w_bar_x = w_bar_x
        self.w_bar_u = w_bar_u

        # 1. Compute feedback gain K_fb
        # Use LQR for K_fb
        self.K_fb, self.P_nominal = compute_LQR_gain(A, B, Q, R, Cx=Cx)

        # 2. Compute Contractivity
        # For projection-based tube, we mainly care about stability in the projected subspace
        # But rigorous tube requires full state stability.
        self.A_cl = A - B @ self.K_fb

        # Check spectral radius for asymptotic stability
        evals = np.linalg.eigvals(self.A_cl)
        self.rho_spec = np.max(np.abs(evals))

        # Check inf-norm for strict box invariance
        self.rho_inf = np.linalg.norm(self.A_cl, np.inf)

        print(f"Tube MPC Design: rho_spec={self.rho_spec:.4f}, rho_inf={self.rho_inf:.4f}")

        # 3. Determine Margins
        # Strategy: Use spectral radius for tightening factor if rho_inf >= 1
        # This is a heuristic for "empirical tube" when strict box invariance fails.
        # Strict Theory: margin = w_bar / (1 - rho_inf)
        # Heuristic: margin = w_bar / (1 - rho_spec) if stable

        if self.rho_inf < 0.999:
            self.tightening_factor = 1.0 / (1.0 - self.rho_inf)
            method = "Strict Inf-Norm"
        elif self.rho_spec < 0.999:
            self.tightening_factor = 1.0 / (1.0 - self.rho_spec)
            method = "Spectral Radius Heuristic"
        else:
            self.tightening_factor = 100.0 # Fallback
            method = "Fallback"
            print("Warning: System barely stable, using large tightening factor.")

        print(f"Tightening Method: {method}, Factor: {self.tightening_factor:.2f}")

        # Compute margins based on projected disturbance bounds
        # margin_x = w_bar_x * factor
        # margin_u = w_bar_u * factor
        self.margin_x = w_bar_x * self.tightening_factor
        self.margin_u = w_bar_u * self.tightening_factor

        # For v: v = v_nom - K_fb e
        # |v - v_nom| <= |K_fb| |e|
        # We need a bound on |e|. If we use component-wise bounds:
        # e_x <= margin_x, e_u <= margin_u.
        # But e has other components.
        # If w_bar_inf is provided, use it to bound full state error norm.
        if w_bar_inf is not None:
             # e_bar_inf = w_bar_inf * factor
             # |K_fb e| <= |K_fb|_inf * e_bar_inf
             self.e_bar_inf = w_bar_inf * self.tightening_factor
             norm_Kfb_inf = np.linalg.norm(self.K_fb, np.inf)
             self.margin_v = norm_Kfb_inf * self.e_bar_inf
        else:
             # Fallback: assume error dominated by x, u components?
             # Or just use a fixed small margin for v if we trust LQR.
             # Let's use 0.0 as placeholder if no info, or small value.
             self.margin_v = 0.1 * (v_max - v_min) # 10% backoff?
             print("Warning: No w_bar_inf provided, using heuristic margin for v.")

        print(f"Margins: x={self.margin_x:.4f}, u={self.margin_u:.4f}, v={self.margin_v:.4f}")

        # 5. Initialize Nominal MPC with tightened constraints
        x_min_tight = x_min + self.margin_x
        x_max_tight = x_max - self.margin_x
        if x_min_tight > x_max_tight:
             print(f"Warning: x constraints tightened to infeasibility! ({x_min_tight:.4f} > {x_max_tight:.4f})")
             # Clamp to center
             center = (x_min + x_max) / 2
             x_min_tight = center - 1e-6
             x_max_tight = center + 1e-6

        u_min_tight = u_min + self.margin_u
        u_max_tight = u_max - self.margin_u
        if u_min_tight > u_max_tight:
             print(f"Warning: u constraints tightened to infeasibility! ({u_min_tight:.4f} > {u_max_tight:.4f})")
             center = (u_min + u_max) / 2
             u_min_tight = center - 1e-6
             u_max_tight = center + 1e-6

        v_min_tight = v_min + self.margin_v
        v_max_tight = v_max - self.margin_v
        if v_min_tight > v_max_tight:
             print(f"Warning: v constraints tightened to infeasibility! ({v_min_tight:.4f} > {v_max_tight:.4f})")
             # If v tightened too much, relax margin_v (it's less critical than state constraints for safety usually)
             # Relax to allow at least small control authority
             v_range = v_max - v_min
             v_min_tight = v_min + 0.45 * v_range
             v_max_tight = v_max - 0.45 * v_range
             print(f"  Relaxed v to 10% range: [{v_min_tight:.4f}, {v_max_tight:.4f}]")

        self.nominal_mpc = LinearMPC(
            A, B, Q, R, self.P_nominal, N,
            x_min_tight, x_max_tight,
            u_min_tight, u_max_tight,
            v_min_tight, v_max_tight,
            Cx, Cu
        )

        self.z_nom = None
        self.d_est = 0.0

    def solve(self, z_current, x_sp):
        if self.z_nom is None:
            self.z_nom = z_current.copy()

        y_meas = self.Cx @ z_current
        y_nom = self.Cx @ self.z_nom
        d_raw = y_meas - y_nom

        alpha = 0.5
        self.d_est = alpha * self.d_est + (1 - alpha) * d_raw

        v_nom_seq, z_nom_seq = self.nominal_mpc.solve(self.z_nom, x_sp, d_est=self.d_est)

        if v_nom_seq is None:
            return None, None

        v_nom = v_nom_seq[0] if v_nom_seq.ndim > 0 else v_nom_seq

        v_feedback = self.K_fb @ (z_current - self.z_nom)
        v_control = v_nom - v_feedback

        self.z_nom = z_nom_seq[1]

        return v_control, z_nom_seq
