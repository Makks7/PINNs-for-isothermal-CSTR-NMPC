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
    def __init__(self, A, B, Q, R, N, x_min, x_max, u_min, u_max, v_min, v_max, Cx, Cu, w_bar):
        """
        w_bar: bound on disturbance |w|_inf <= w_bar
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
        self.w_bar = w_bar

        # 1. Compute feedback gain K_fb for error system
        # First try LQR
        self.K_fb, self.P_nominal = compute_LQR_gain(A, B, Q, R, Cx=Cx)

        # Check rho with LQR
        # A_cl = A - B K_fb
        self.A_cl = A - B @ self.K_fb
        self.rho = np.linalg.norm(self.A_cl, np.inf)
        print(f"LQR rho (inf-norm) = {self.rho:.4f}")

        # If LQR fails to be contractive in inf-norm, try to optimize K_fb
        if self.rho >= 0.95: # Threshold bit lower than 1 to be safe
            print("LQR gain not contractive enough. Optimizing K_fb for inf-norm...")
            K_inf, rho_inf = compute_inf_norm_gain(A, B)

            if K_inf is not None and rho_inf < 1.0:
                print(f"Found optimized gain with rho={rho_inf:.4f}")
                self.K_fb = K_inf
                self.rho = rho_inf
                self.A_cl = A - B @ self.K_fb
            else:
                print(f"Failed to find contractive gain (rho={rho_inf if rho_inf else 'None'}). Using LQR fallback.")
                if self.rho >= 1.0:
                    self.rho = 0.99

        # 3. Compute invariant set size
        if self.rho < 0.999:
            self.e_bar = w_bar / (1.0 - self.rho)
        else:
            self.e_bar = w_bar * 1000

        # 4. Compute margins for tightening
        norm_Cx_1 = np.linalg.norm(Cx, 1)
        self.margin_x = norm_Cx_1 * self.e_bar

        norm_Cu_1 = np.linalg.norm(Cu, 1)
        self.margin_u = norm_Cu_1 * self.e_bar

        norm_Kfb_inf = np.linalg.norm(self.K_fb, np.inf)
        self.margin_v = norm_Kfb_inf * self.e_bar

        print(f"Tube MPC Initialized: rho={self.rho:.4f}, e_bar={self.e_bar:.4f}")
        print(f"Margins: x={self.margin_x:.4f}, u={self.margin_u:.4f}, v={self.margin_v:.4f}")

        # 5. Initialize Nominal MPC with tightened constraints
        x_min_tight = x_min + self.margin_x
        x_max_tight = x_max - self.margin_x
        if x_min_tight > x_max_tight:
             print(f"Warning: x constraints tightened to infeasibility! ({x_min_tight:.2f} > {x_max_tight:.2f})")

        u_min_tight = u_min + self.margin_u
        u_max_tight = u_max - self.margin_u
        if u_min_tight > u_max_tight:
             print(f"Warning: u constraints tightened to infeasibility! ({u_min_tight:.2f} > {u_max_tight:.2f})")

        v_min_tight = v_min + self.margin_v
        v_max_tight = v_max - self.margin_v
        if v_min_tight > v_max_tight:
             print(f"Warning: v constraints tightened to infeasibility! ({v_min_tight:.2f} > {v_max_tight:.2f})")

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
