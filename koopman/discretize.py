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

def compute_empirical_bound(A, B, Z, V, Z_next, margin_ratio=1.0):
    """
    Compute empirical w_bar based on one-step prediction errors on validation data.
    w_k = z_{k+1} - (A z_k + B v_k)
    w_bar = max_k |w_k|_inf
    """
    # Predict Z_next
    # Z: (N, nz), V: (N,), B: (nz, 1) -> V.reshape(-1, 1) @ B.T -> (N, nz)
    # A: (nz, nz) -> Z @ A.T -> (N, nz)
    Z_pred = Z @ A.T + V.reshape(-1, 1) @ B.T

    residuals = Z_next - Z_pred

    # Ignore constant state (last column usually, or index of '1')
    # Since we don't know index here easily without lifting object,
    # we assume the user handles it or we compute norm over all.
    # But constant state error should be 0 by construction/definition.
    # If not 0, it contributes to error.

    # Compute infinity norm for each sample
    # |w_k|_inf = max_i |w_k[i]|
    w_norms = np.linalg.norm(residuals, ord=np.inf, axis=1)

    w_bar = np.max(w_norms) * margin_ratio
    return w_bar
