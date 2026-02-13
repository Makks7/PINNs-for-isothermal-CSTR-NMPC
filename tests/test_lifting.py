import unittest
import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from koopman.lifting import KoopmanLifting

class TestLifting(unittest.TestCase):
    def test_psi_dot(self):
        # Parameters
        dt = 1e-6
        lifting = KoopmanLifting(degree=3)

        # Random states
        N = 10
        X = np.random.rand(N)
        U = np.random.rand(N)
        V = np.random.rand(N)

        # Calculate analytical Z_dot
        Z_dot_analytical = lifting.psi_dot(X, U, V)

        # Calculate numerical Z_dot
        # \dot x = u(C_Ai - x) - k x
        dx_dt = U * (lifting.C_Ai - X) - lifting.k * X
        # \dot u = v
        du_dt = V

        X_next = X + dx_dt * dt
        U_next = U + du_dt * dt

        Z_curr = lifting.lift(X, U)
        Z_next = lifting.lift(X_next, U_next)

        Z_dot_numerical = (Z_next - Z_curr) / dt

        # Check if they are close
        error = np.abs(Z_dot_analytical - Z_dot_numerical)
        max_error = np.max(error)

        print(f"Max error between analytical and numerical derivative: {max_error}")
        self.assertTrue(max_error < 1e-4, f"Derivative error too high: {max_error}")

if __name__ == '__main__':
    unittest.main()
