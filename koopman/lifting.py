import numpy as np

class KoopmanLifting:
    def __init__(self, degree=2, C_Ai=1.0, k=0.028):
        self.degree = degree
        self.C_Ai = C_Ai
        self.k = k
        self.powers = self._build_monomial_powers(degree)
        self.n_z = len(self.powers)
        self.feature_names = [f"x^{p[0]}u^{p[1]}" for p in self.powers]

    def _build_monomial_powers(self, d):
        """
        Returns list of (i, j) tuples such that i + j <= d.
        Sorted by total degree, then by x power (descending).
        """
        powers = []
        for total_deg in range(d + 1):
            # For each degree, sort by x power descending: (deg, 0), (deg-1, 1), ..., (0, deg)
            for i in range(total_deg, -1, -1):
                j = total_deg - i
                powers.append((i, j))
        return powers

    def get_feature_index(self, i, j):
        try:
            return self.powers.index((i, j))
        except ValueError:
            return None

    def lift(self, X, U):
        """
        Lift state x and input u to z space.
        Args:
            X: (N,) array or scalar
            U: (N,) array or scalar
        Returns:
            Z: (N, n_z) array
        """
        X = np.atleast_1d(X).flatten()
        U = np.atleast_1d(U).flatten()

        if len(X) != len(U):
            raise ValueError("X and U must have same length")

        N = len(X)
        Z = np.zeros((N, self.n_z))

        for idx, (i, j) in enumerate(self.powers):
            # Compute x^i * u^j
            if i == 0 and j == 0:
                Z[:, idx] = 1.0
            elif i == 0:
                Z[:, idx] = U**j
            elif j == 0:
                Z[:, idx] = X**i
            else:
                Z[:, idx] = (X**i) * (U**j)

        return Z

    def lift_one(self, x, u):
        return self.lift(np.array([x]), np.array([u]))[0]

    def project(self, z, Cx=None, Cu=None):
        z = np.asarray(z).flatten()

        # Determine x and u from z
        # Method 1: Use Cx, Cu if provided
        if Cx is not None:
            x = float(np.dot(Cx, z))
        else:
            # Fallback: Assume index 1 is x (based on current ordering for d>=1)
            # x index is (1,0). get_feature_index(1,0) should return correct index.
            idx_x = self.get_feature_index(1, 0)
            if idx_x is not None:
                x = z[idx_x]
            else:
                x = 0.0 # Should not happen

        if Cu is not None:
            u = float(np.dot(Cu, z))
        else:
            # Fallback: Assume index 2 is u (based on current ordering for d>=1)
            idx_u = self.get_feature_index(0, 1)
            if idx_u is not None:
                u = z[idx_u]
            else:
                u = 0.0

        return self.lift_one(x, u)

    def psi_dot(self, X, U, V):
        """
        Analytical time derivative of lifted state z.
        \dot{z} = \frac{d}{dt} \psi(x, u)
        Args:
            X, U, V: (N,) arrays
        Returns:
            Z_dot: (N, n_z) array
        """
        X = np.atleast_1d(X).flatten()
        U = np.atleast_1d(U).flatten()
        V = np.atleast_1d(V).flatten()

        N = len(X)
        Z_dot = np.zeros((N, self.n_z))

        # Physics:
        # \dot x = u(C_Ai - x) - k x
        dx_dt = U * (self.C_Ai - X) - self.k * X
        # \dot u = v
        du_dt = V

        for idx, (i, j) in enumerate(self.powers):
            # \frac{d}{dt}(x^i u^j) = i x^{i-1} u^j \dot x + j x^i u^{j-1} \dot u

            term1 = np.zeros(N)
            term2 = np.zeros(N)

            # Term 1: derivative wrt x -> i * x^(i-1) * u^j * dx_dt
            if i > 0:
                if i == 1:
                    d_x_part = np.ones(N)
                else:
                    d_x_part = X**(i-1)

                if j > 0:
                    d_x_part *= (U**j)

                term1 = i * d_x_part * dx_dt

            # Term 2: derivative wrt u -> j * x^i * u^(j-1) * du_dt
            if j > 0:
                if j == 1:
                    d_u_part = np.ones(N)
                else:
                    d_u_part = U**(j-1)

                if i > 0:
                    d_u_part *= (X**i)

                term2 = j * d_u_part * du_dt # du_dt is V

            Z_dot[:, idx] = term1 + term2

        return Z_dot

    def get_output_matrices(self):
        """
        Returns Cx and Cu such that x = Cx z, u = Cu z
        """
        Cx = np.zeros(self.n_z)
        Cu = np.zeros(self.n_z)

        idx_x = self.get_feature_index(1, 0)
        idx_u = self.get_feature_index(0, 1)

        if idx_x is not None:
            Cx[idx_x] = 1.0
        if idx_u is not None:
            Cu[idx_u] = 1.0

        return Cx, Cu
