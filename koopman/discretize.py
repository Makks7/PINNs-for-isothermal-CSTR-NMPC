import numpy as np
from scipy.linalg import expm

def discretize_generator(K, L, dt):
    """
    Compute A = exp(K*dt) and B = integral(exp(K*tau)*L) dtau
    using block matrix exponential.
    """
    n_z = K.shape[0]
    n_u = L.shape[1]

    # Form block matrix
    # [K L]
    # [0 0]
    M = np.zeros((n_z + n_u, n_z + n_u))
    M[:n_z, :n_z] = K
    M[:n_z, n_z:] = L

    # Matrix exponential
    ExpM = expm(M * dt)

    # Extract A and B
    A = ExpM[:n_z, :n_z]
    B = ExpM[:n_z, n_z:]

    return A, B

def compute_theoretical_bound(K, dt, max_residual_inf_norm):
    """
    Compute w_bar based on continuous residual epsilon.
    |w_k|_inf <= epsilon_bar * (exp(|K|_inf * dt) - 1) / |K|_inf
    """
    norm_K_inf = np.linalg.norm(K, np.inf)

    if np.abs(norm_K_inf) < 1e-9:
        scaling_factor = dt
    else:
        scaling_factor = (np.exp(norm_K_inf * dt) - 1.0) / norm_K_inf

    w_bar = max_residual_inf_norm * scaling_factor
    return w_bar

def compute_empirical_bound(A, B, Z, V, Z_next, Cx, Cu, margin_ratio=1.0):
    """
    Compute empirical w_bar based on one-step prediction errors on validation data.
    w_k = z_{k+1} - (A z_k + B v_k)
    Returns:
        w_bar_x: max |Cx w_k|
        w_bar_u: max |Cu w_k|
        w_bar_v_est: max |w_k|_inf (full state bound, potentially large)
    """
    # Predict Z_next
    # Z: (N, nz), V: (N,), B: (nz, 1) -> V.reshape(-1, 1) @ B.T -> (N, nz)
    # A: (nz, nz) -> Z @ A.T -> (N, nz)
    Z_pred = Z @ A.T + V.reshape(-1, 1) @ B.T

    residuals = Z_next - Z_pred

    # Project residuals
    w_x = residuals @ Cx
    w_u = residuals @ Cu

    # Compute bounds (99th percentile for robustness against outliers)
    w_bar_x = np.percentile(np.abs(w_x), 99) * margin_ratio
    w_bar_u = np.percentile(np.abs(w_u), 99) * margin_ratio

    # Full state bound for reference (or conservative tube calculation)
    w_norms = np.linalg.norm(residuals, ord=np.inf, axis=1)
    w_bar_inf = np.percentile(w_norms, 99) * margin_ratio

    return w_bar_x, w_bar_u, w_bar_inf
