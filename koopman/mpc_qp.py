import cvxpy as cp
import numpy as np
from scipy.linalg import solve_discrete_are

def compute_LQR_gain(A, B, Q, R, Cx=None):
    # Construct Q_z for the lifted state
    if Cx is not None:
        if np.isscalar(Q):
            Q_z = Q * np.outer(Cx, Cx)
        else:
            Q_z = Cx.T @ Q @ Cx
    else:
        if np.isscalar(Q):
            Q_z = Q * np.eye(A.shape[0])
        else:
            Q_z = Q

    # Ensure R is 2D
    if np.isscalar(R):
        R_mat = np.array([[R]])
    else:
        R_mat = R

    # Solve DARE
    # Hack: Scale A slightly to move eigenvalues off unit circle if needed
    evals = np.linalg.eigvals(A)
    if np.any(np.abs(np.abs(evals) - 1.0) < 1e-4):
        # print("Warning: Eigenvalues on unit circle. Scaling A for LQR design.")
        A_design = A * 0.995
    else:
        A_design = A

    P = solve_discrete_are(A_design, B, Q_z, R_mat)

    # Compute K
    R_plus_BTPB = R_mat + B.T @ P @ B
    K = np.linalg.inv(R_plus_BTPB) @ (B.T @ P @ A)

    return K, P

class LinearMPC:
    def __init__(self, A, B, Q, R, P, N, x_min, x_max, u_min, u_max, v_min, v_max, Cx, Cu):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P
        self.N = N
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        self.v_min = v_min
        self.v_max = v_max
        self.Cx = Cx
        self.Cu = Cu

        self.nz = A.shape[0]
        self.nu = B.shape[1] if len(B.shape) > 1 else 1

        self.C_Ai_val = 1.0
        self.k_val = 0.028

    def solve(self, z0, x_sp, d_est=0.0):
        """
        Solve MPC with optional disturbance estimate d_est.
        Model: x = Cx z + d_est
        Tracking: x -> x_sp  =>  Cx z -> x_sp - d_est
        Constraints: x_min <= Cx z + d_est <= x_max
        """
        Z = cp.Variable((self.N + 1, self.nz))
        V = cp.Variable((self.N, self.nu))

        cost = 0
        constraints = [Z[0] == z0]

        # Effective setpoint for the physical model part
        x_sp_eff = x_sp - d_est

        # Calculate steady state for tracking x_sp_eff
        # 0 = u(1-x) - kx -> u = kx/(1-x)
        if np.abs(self.C_Ai_val - x_sp_eff) > 1e-6:
             u_ss = self.k_val * x_sp_eff / (self.C_Ai_val - x_sp_eff)
        else:
             u_ss = 0.0

        z_ss = np.zeros(self.nz)
        # We need to know feature mapping to construct z_ss correctly.
        # This class assumes a specific lifting structure if we hardcode indices.
        # Ideally, we should use a method from KoopmanLifting to invert/construct.
        # But here we can use a heuristic or pass z_ss from outside.
        # For now, let's assume the basic lifting [x, u, ...] is always present at indices 0, 1?
        # No, KoopmanLifting order depends on degree.
        # Wait, get_feature_index logic is in KoopmanLifting.
        # LinearMPC doesn't know about KoopmanLifting instance.
        # Use Cx, Cu to identify indices?
        # Cx has 1.0 at x index.
        idx_x = np.argmax(np.abs(self.Cx))
        idx_u = np.argmax(np.abs(self.Cu))

        z_ss[idx_x] = x_sp_eff
        z_ss[idx_u] = u_ss

        # We can't easily fill higher order terms (xu, x^2) without lifting function.
        # However, for Regulation, we usually penalize x deviation.
        # If P matrix is consistent, we should compute z_ss correctly.
        # Or we only penalize x and u deviations in terminal cost?
        # Or we rely on the fact that if x, u are close, z is close.
        # Let's fill what we can.
        # If we have P from LQR, it penalizes z.

        for k in range(self.N):
            # Stage cost: Q(x - x_sp)^2 + R v^2
            # x_k (measured) = Cx z_k + d_est
            # Error = (Cx z_k + d_est) - x_sp = Cx z_k - (x_sp - d_est) = Cx z_k - x_sp_eff
            x_k_model = self.Cx @ Z[k]
            cost += self.Q * cp.sum_squares(x_k_model - x_sp_eff) + self.R * cp.sum_squares(V[k])

            # Dynamics
            constraints.append(Z[k+1] == self.A @ Z[k] + self.B @ V[k])

            # Constraints
            u_k = self.Cu @ Z[k]

            constraints.append(u_k >= self.u_min)
            constraints.append(u_k <= self.u_max)

            # x constraints: x_min <= Cx z + d_est <= x_max
            # => x_min - d_est <= Cx z <= x_max - d_est
            constraints.append(x_k_model >= self.x_min - d_est)
            constraints.append(x_k_model <= self.x_max - d_est)

            constraints.append(V[k] >= self.v_min)
            constraints.append(V[k] <= self.v_max)

        # Terminal cost
        # We use z_ss as target.
        # Note: z_ss is incomplete (higher order terms are 0).
        # This might bias the terminal cost.
        # But typically terminal cost weight on higher order terms is small if Q only penalizes x.
        # Q_z = Cx^T Q Cx. P comes from Q_z.
        # If Q only on x, P handles z such that x is regulated.
        # So deviation in 'xu' is only penalized insofar as it affects future 'x'.
        cost += cp.quad_form(Z[self.N] - z_ss, self.P)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None, None

        return V[0].value, Z.value
