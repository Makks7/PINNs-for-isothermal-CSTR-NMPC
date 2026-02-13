import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import json
import time

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from koopman import KoopmanLifting, KoopmanGenerator, discretize_generator
from koopman.discretize import compute_empirical_bound
from koopman.tube_mpc import TubeMPC
from scipy.integrate import odeint

# CSTR Dynamics for simulation (Ground Truth)
def cstr_dynamics(state, t, v_in):
    C_Ai = 1.0
    k = 0.028
    x_val, u_val = state
    dxdt = u_val * (C_Ai - x_val) - k * x_val
    dudt = v_in
    return [dxdt, dudt]

def train_and_select_model(degrees=[2, 3], lambdas=[1e-5]):
    print("Starting Model Selection...")

    # Load data
    df = pd.read_csv('koopman_data.csv')

    # Split train/val
    traj_ids = df['traj_id'].unique()
    np.random.seed(42)
    np.random.shuffle(traj_ids)

    n_train = int(0.8 * len(traj_ids))
    train_ids = traj_ids[:n_train]
    val_ids = traj_ids[n_train:]

    df_train = df[df['traj_id'].isin(train_ids)]
    df_val = df[df['traj_id'].isin(val_ids)]

    df_train.to_csv('koopman_train.csv', index=False)

    best_model = None
    best_rmse = float('inf')
    best_params = {}

    for d in degrees:
        for lam in lambdas:
            print(f"Training degree={d}, lambda={lam}...")
            generator = KoopmanGenerator('koopman_train.csv', degree=d, lambda_reg=lam)
            K, L = generator.fit()

            dt = df_val['dt'].iloc[0]
            A, B = discretize_generator(K, L, dt)

            X_val = df_val['x_k'].values
            U_val = df_val['u_k'].values
            V_val = df_val['v_k'].values
            X_next_val = df_val['x_next'].values
            U_next_val = df_val['u_next'].values

            Z_val = generator.lifting.lift(X_val, U_val)
            Z_next_val_true = generator.lifting.lift(X_next_val, U_next_val)

            Z_pred = Z_val @ A.T + V_val.reshape(-1, 1) @ B.T

            idx_x = generator.lifting.get_feature_index(1, 0)
            if idx_x is not None:
                rmse_x = np.sqrt(np.mean((Z_next_val_true[:, idx_x] - Z_pred[:, idx_x])**2))
            else:
                rmse_x = np.mean((Z_next_val_true - Z_pred)**2)

            print(f"  RMSE (x): {rmse_x:.6f}")

            if rmse_x < best_rmse:
                best_rmse = rmse_x
                best_model = (generator, K, L, A, B)
                best_params = {'degree': d, 'lambda': lam}

    print(f"Best Model: degree={best_params['degree']}, lambda={best_params['lambda']}, RMSE={best_rmse:.6f}")

    return best_model, best_params

def run_experiment():
    dt = 0.1

    if not os.path.exists('koopman_data.csv'):
        print("Error: koopman_data.csv not found.")
        return

    (generator, K, L, A, B), params = train_and_select_model(degrees=[2, 3], lambdas=[1e-5])

    # Compute Component-wise Error Bounds
    df = pd.read_csv('koopman_data.csv')
    X = df['x_k'].values
    U = df['u_k'].values
    V = df['v_k'].values
    X_next = df['x_next'].values
    U_next = df['u_next'].values

    Z = generator.lifting.lift(X, U)
    Z_next = generator.lifting.lift(X_next, U_next)

    Cx, Cu = generator.lifting.get_output_matrices()

    # Use output residuals for w_bar
    Z_pred = Z @ A.T + V.reshape(-1, 1) @ B.T

    x_pred = Z_pred @ Cx
    u_pred = Z_pred @ Cu
    res_x = X_next - x_pred
    res_u = U_next - u_pred

    # Use 80th percentile for less conservative bounds as per prompt suggestion
    w_bar_x = np.percentile(np.abs(res_x), 80)
    w_bar_u = np.percentile(np.abs(res_u), 80)

    print(f"Empirical w_bar_x (80%): {w_bar_x:.6f}")
    print(f"Empirical w_bar_u (80%): {w_bar_u:.6f}")

    # Setup MPC
    x_min, x_max = 0.0, 1.0
    u_min, u_max = 0.0, 3.0
    v_min, v_max = -2.0, 2.0

    Q = 100.0
    R = 0.1
    N = 20

    print("Initializing Tube MPC...")
    mpc = TubeMPC(
        A, B, Q, R, N,
        x_min, x_max, u_min, u_max, v_min, v_max,
        Cx, Cu, w_bar_x, w_bar_u, lifting=generator.lifting
    )

    # Simulation
    T_sim = 50.0
    n_steps = int(T_sim / dt)
    t_eval = np.linspace(0, T_sim, n_steps+1)

    x0 = 0.2
    u0 = 0.5
    current_state = [x0, u0]

    state_hist = []
    input_hist = []
    x_sp = 0.8

    start_time = time.time()

    print(f"Starting simulation: x0={x0}, x_sp={x_sp}...")

    infeasible_count = 0

    for i in range(n_steps):
        z_curr = generator.lifting.lift([current_state[0]], [current_state[1]])[0]

        v_opt, z_pred = mpc.solve(z_curr, x_sp)

        if v_opt is None:
            infeasible_count += 1
            if infeasible_count <= 5:
                print(f"MPC Infeasible at step {i}")
            v_apply = 0.0
        else:
            v_apply = v_opt if np.isscalar(v_opt) else v_opt[0]

        t_span = [0, dt]
        sol = odeint(cstr_dynamics, current_state, t_span, args=(v_apply,))
        next_state = sol[-1]

        state_hist.append(current_state)
        input_hist.append(v_apply)

        current_state = next_state

    end_time = time.time()
    avg_time = (end_time - start_time) / n_steps
    print(f"Simulation finished. Avg time per step: {avg_time*1000:.2f} ms")
    print(f"Total Infeasible Steps: {infeasible_count}/{n_steps}")

    state_hist = np.array(state_hist)
    input_hist = np.array(input_hist)

    x_traj = state_hist[:, 0]
    iae = np.sum(np.abs(x_traj - x_sp)) * dt
    energy = np.sum(input_hist**2) * dt

    print(f"IAE: {iae:.4f}")
    print(f"Control Energy: {energy:.4f}")
    print(f"Final u: {state_hist[-1, 1]:.4f}")
    print(f"Target u_ss: {0.028*0.8/0.2:.4f}")

    results_df = pd.DataFrame({
        'time': t_eval[:-1],
        'x': x_traj,
        'u': state_hist[:, 1],
        'v': input_hist,
        'x_sp': x_sp
    })
    results_df.to_csv('experiments/koopman_mpc_data.csv', index=False)

    config = {
        'degree': params['degree'],
        'lambda': params['lambda'],
        'w_bar_x': w_bar_x,
        'rho_spec': mpc.rho_spec,
        'tightening_factor': mpc.tightening_factor,
        'margin_x': mpc.margin_x,
        'margin_u': mpc.margin_u,
        'IAE': iae,
        'Energy': energy,
        'dt': dt,
        'N': N
    }
    with open('experiments/run_metadata.json', 'w') as f:
        json.dump(config, f, indent=4)

    plt.figure(figsize=(10, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t_eval[:-1], x_traj, label='x (C_A)', linewidth=2)
    plt.axhline(x_sp, color='r', linestyle='--', label='Setpoint')
    plt.ylabel('Concentration')
    plt.legend()
    plt.grid()
    plt.title(f"Koopman Tube MPC (d={params['degree']})")

    plt.subplot(3, 1, 2)
    plt.plot(t_eval[:-1], state_hist[:, 1], label='u (Dilution)')
    plt.axhline(u_min, color='k', linestyle=':')
    plt.axhline(u_max, color='k', linestyle=':')
    plt.ylabel('Input u')
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t_eval[:-1], input_hist, label='v (Delta u)')
    plt.axhline(v_min, color='k', linestyle=':')
    plt.axhline(v_max, color='k', linestyle=':')
    plt.ylabel('Input Rate v')
    plt.grid()

    plt.tight_layout()
    plt.savefig('experiments/koopman_mpc_result.png')
    print("Results saved to experiments/koopman_mpc_result.png")

if __name__ == "__main__":
    run_experiment()
