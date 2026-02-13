import numpy as np
import cvxpy as cp
import pandas as pd
from .lifting import KoopmanLifting

class KoopmanGenerator:
    def __init__(self, data_path, degree=2, lambda_reg=1e-5):
        self.data_path = data_path
        self.degree = degree
        self.lambda_reg = lambda_reg
        self.lifting = KoopmanLifting(degree=degree)

        # System parameters (for constraints)
        self.k = self.lifting.k
        self.C_Ai = self.lifting.C_Ai

    def fit(self):
        # 1. Load data
        df = pd.read_csv(self.data_path)

        X = df['x_k'].values
        U = df['u_k'].values
        V = df['v_k'].values

        n_samples = len(X)

        # 2. Compute lifted state Z and its derivative Z_dot
        Z = self.lifting.lift(X, U)
        Z_dot = self.lifting.psi_dot(X, U, V)

        # 3. Formulate optimization problem
        n_z = self.lifting.n_z
        K = cp.Variable((n_z, n_z))
        L = cp.Variable((n_z, 1))

        # Prediction: \dot{Z} = Z @ K.T + V @ L.T
        # Note: Z is (N, nz). K is (nz, nz). Z @ K.T is (N, nz).
        # V is (N,). L is (nz, 1). V.reshape(-1, 1) @ L.T is (N, 1) @ (1, nz) -> (N, nz).
        prediction = Z @ K.T + V.reshape(-1, 1) @ L.T

        # Loss: ||Z_dot - prediction||^2 + lambda * ||K||^2
        loss = cp.sum_squares(Z_dot - prediction) + self.lambda_reg * cp.sum_squares(K)

        constraints = []

        # 4. Add Physics Constraints

        # Constraint for x: \dot x = -k x + C_Ai u - xu
        idx_x = self.lifting.get_feature_index(1, 0)
        idx_u = self.lifting.get_feature_index(0, 1)
        idx_xu = self.lifting.get_feature_index(1, 1)
        idx_1 = self.lifting.get_feature_index(0, 0)

        if idx_x is not None:
            # Row x of K
            constraints.append(K[idx_x, idx_x] == -self.k)
            if idx_u is not None:
                constraints.append(K[idx_x, idx_u] == self.C_Ai)
            if idx_xu is not None:
                constraints.append(K[idx_x, idx_xu] == -1.0)

            # All other elements in row x of K should be 0?
            # Strictly speaking, yes, if the model is exact.
            # But maybe we allow small non-zero terms for other basis functions if they help?
            # The prompt says "hard structure".
            # "Row 1 (x) forced mass conservation".
            # This implies the equation is exact.
            # So all other coefficients must be 0.

            # Construct a mask of indices that are allowed to be non-zero
            allowed_indices = [idx for idx in [idx_x, idx_u, idx_xu] if idx is not None]
            for j in range(n_z):
                if j not in allowed_indices:
                    constraints.append(K[idx_x, j] == 0.0)

            # Row x of L should be 0 (no direct effect of v on x)
            for j in range(1):
                constraints.append(L[idx_x, j] == 0.0)

        # Constraint for u: \dot u = v
        if idx_u is not None:
            # Row u of K should be all 0
            for j in range(n_z):
                constraints.append(K[idx_u, j] == 0.0)
            # Row u of L should be 1
            constraints.append(L[idx_u, 0] == 1.0)

        # Constraint for 1: \dot 1 = 0
        if idx_1 is not None:
            # Row 1 of K and L should be all 0
            for j in range(n_z):
                constraints.append(K[idx_1, j] == 0.0)
            constraints.append(L[idx_1, 0] == 0.0)

        # Solve
        prob = cp.Problem(cp.Minimize(loss), constraints)
        prob.solve(solver=cp.OSQP, verbose=False) # OSQP is usually robust

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Warning: Optimization status: {prob.status}")
            # Try SCS if OSQP fails?
            if prob.status != cp.OPTIMAL:
                 try:
                     prob.solve(solver=cp.SCS, verbose=False)
                     print(f"SCS status: {prob.status}")
                 except:
                     pass

        self.K_matrix = K.value
        self.L_matrix = L.value

        # Calculate residual
        # Using continuous time residual: Z_dot - (K Z + L v)
        residual = Z_dot - (Z @ self.K_matrix.T + V.reshape(-1, 1) @ self.L_matrix.T)
        self.max_residual = np.max(np.linalg.norm(residual, axis=1))

        return self.K_matrix, self.L_matrix
