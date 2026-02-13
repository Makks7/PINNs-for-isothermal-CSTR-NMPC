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
                 w_bar_x, w_bar_u, w_bar_inf=None, lifting=None):
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
        self.lifting = lifting

        # 1. Compute feedback gain K_fb for error system
        # First try LQR
        self.K_fb, self.P_nominal = compute_LQR_gain(A, B, Q, R, Cx=Cx)

        # Extract Ky from K_fb (assuming first two components are x and u)
        # K_fb is (nu, nz). We want feedback on y=[x, u].
        # Assuming x is index 1, u is index 2 in lifting order (checked previously: x=1, u=2).
        # Wait, if K_fb is trained on z, we need to extract the parts corresponding to x and u?
        # Or do we mean "feedback on physical outputs"?
        # The prompt says: "Minimum implementation: take x, u components from K_fb".
        # But indices depend on lifting.
        # However, for now let's assume indices are correct or use Cx/Cu to project K_fb?
        # K_fb z = K_fb @ (P_x x + P_u u + ...)
        # If we only feedback on x, u, we can try to approximate K_fb z ~ K_x x + K_u u.
        # But better to just take the coefficients of x and u in K_fb.

        # Determine indices
        if self.lifting:
            idx_x = self.lifting.get_feature_index(1, 0)
            idx_u = self.lifting.get_feature_index(0, 1)
        else:
            # Fallback indices (assuming standard order)
            idx_x = 1
            idx_u = 2

        # Ky construction:
        # K_fb is (1, nz). Ky should be (1, 2) acting on [x, u].
        # Ky = [K_fb[0, idx_x], K_fb[0, idx_u]]
        self.Ky = np.array([[self.K_fb[0, idx_x], self.K_fb[0, idx_u]]])

        # 2. Compute Contractivity
        self.A_cl = A - B @ self.K_fb
        evals = np.linalg.eigvals(self.A_cl)
        self.rho_spec = np.max(np.abs(evals))
        self.rho_inf = np.linalg.norm(self.A_cl, np.inf)

        print(f"Tube MPC Design: rho_spec={self.rho_spec:.4f}, rho_inf={self.rho_inf:.4f}")

        # 3. Determine Margins
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

        self.margin_x = w_bar_x * self.tightening_factor
        self.margin_u = w_bar_u * self.tightening_factor

        # For v: Temporarily set margin_v to 0.0 to avoid locking up
        self.margin_v = 0.0

        print(f"Margins: x={self.margin_x:.4f}, u={self.margin_u:.4f}, v={self.margin_v:.4f}")

        # 5. Initialize Nominal MPC with tightened constraints
        x_min_tight = x_min + self.margin_x
        x_max_tight = x_max - self.margin_x
        if x_min_tight > x_max_tight:
             print(f"Warning: x constraints tightened to infeasibility! ({x_min_tight:.4f} > {x_max_tight:.4f})")
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

        self.nominal_mpc = LinearMPC(
            A, B, Q, R, self.P_nominal, N,
            x_min_tight, x_max_tight,
            u_min_tight, u_max_tight,
            v_min_tight, v_max_tight,
            Cx, Cu, lifting=self.lifting
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

        # Feedback on physical subspace
        # y_curr = [x, u], y_nom = [x_nom, u_nom]
        y_curr = np.array([float(self.Cx @ z_current), float(self.Cu @ z_current)])
        y_nom_phys  = np.array([float(self.Cx @ self.z_nom), float(self.Cu @ self.z_nom)])

        v_control = float(v_nom - (self.Ky @ (y_curr - y_nom_phys))[0])

        # Clip v
        v_control = float(np.clip(v_control, self.v_min, self.v_max))

        # Update z_nom with PROJECTION
        # z_nom_seq[1] is the predicted next nominal state by linear model
        # We project it back to the manifold
        if self.lifting is not None:
            self.z_nom = self.lifting.project(z_nom_seq[1], Cx=self.Cx, Cu=self.Cu)
        else:
            self.z_nom = z_nom_seq[1]

        return v_control, z_nom_seq
