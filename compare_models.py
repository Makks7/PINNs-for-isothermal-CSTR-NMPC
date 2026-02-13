# compare_models.py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

from koopman import KoopmanGenerator, discretize_generator

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
CSV_FILE = 'cstr_simulation_data.csv'
PINN_MODEL = 'train_results/trained_model.pth'
VANILLA_MODEL = 'train_results_vanilla/vanilla_model.pth'
METADATA_FILE = 'experiments/run_metadata.json'

SAVE_DIR = 'comparison_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# ────────────────────────────────────────────────
# MODEL DEFINITIONS (Placeholder)
# ────────────────────────────────────────────────
class PINN_CSTR(torch.nn.Module):
    def __init__(self, hidden_layers=4, neurons=64):
        super().__init__()
        layers = [torch.nn.Linear(3, neurons), torch.nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([torch.nn.Linear(neurons, neurons), torch.nn.Tanh()])
        layers.append(torch.nn.Linear(neurons, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, C_A0, u, t):
        x = torch.cat([C_A0, u, t], dim=1)
        return self.network(x)

class Vanilla_NN(torch.nn.Module):
    def __init__(self, hidden_layers=4, neurons=64):
        super().__init__()
        layers = [torch.nn.Linear(3, neurons), torch.nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([torch.nn.Linear(neurons, neurons), torch.nn.Tanh()])
        layers.append(torch.nn.Linear(neurons, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, C_A0, u, t):
        x = torch.cat([C_A0, u, t], dim=1)
        return self.network(x)

# ────────────────────────────────────────────────
# ROLLOUT PREDICTION
# ────────────────────────────────────────────────
def rollout_prediction(model, df):
    t = df['time'].values
    u = df['u_input'].values
    CA_true = df['CA_concentration'].values

    CA_pred = np.zeros_like(CA_true)
    CA_pred[0] = CA_true[0]
    CA_current = CA_true[0]

    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        u_now = u[i-1]

        with torch.no_grad():
            inp = torch.tensor([[CA_current, u_now, dt]], dtype=torch.float32)
            CA_next = model(inp[:,0:1], inp[:,1:2], inp[:,2:3]).item()

        CA_pred[i] = CA_next
        CA_current = CA_next

    return CA_pred, t

def predict_koopman(df, degree=3, lambda_reg=1e-5):
    print(f"Fitting Koopman model (d={degree}, lam={lambda_reg}) for comparison...")
    if not os.path.exists('koopman_data.csv'):
        print("koopman_data.csv not found, skipping Koopman.")
        return None

    generator = KoopmanGenerator('koopman_data.csv', degree=degree, lambda_reg=lambda_reg)
    K, L = generator.fit()

    t = df['time'].values
    u = df['u_input'].values
    CA_true = df['CA_concentration'].values

    # Compute v (input rate)
    v = np.zeros_like(u)
    dt_vals = np.diff(t)
    # Forward difference for v? Or backward?
    # u[k+1] = u[k] + v[k]*dt
    # v[k] = (u[k+1] - u[k])/dt
    # We use v[k] to predict x[k+1].
    v[:-1] = np.diff(u) / dt_vals
    v[-1] = 0.0 # No input change at last step

    preds = np.zeros_like(CA_true)
    preds[0] = CA_true[0]

    z_curr = generator.lifting.lift(CA_true[0], u[0])[0]

    idx_x = generator.lifting.get_feature_index(1, 0)

    for i in range(len(t)-1):
        dt = dt_vals[i]
        A, B = discretize_generator(K, L, dt)

        v_k = v[i]

        # z_next = A z + B v
        z_next = A @ z_curr + B.flatten() * v_k

        # Extract x prediction
        if idx_x is not None:
            preds[i+1] = z_next[idx_x]
        else:
             # Fallback
             preds[i+1] = z_next[0] # assuming x is first? Not guaranteed.
             # Wait, get_feature_index(1,0) should exist if degree>=1.

        # Open Loop Feedback Correction:
        # We are doing open-loop rollout for x.
        # But u is GIVEN by the dataset.
        # Koopman state contains u.
        # We should overwrite the u-component of z_next with the TRUE u from dataset?
        # Standard "simulation" mode uses predicted x, but GIVEN u.
        # Yes.

        # Re-lift based on predicted x and TRUE u
        u_next_true = u[i+1]
        x_next_pred = preds[i+1]

        z_curr = generator.lifting.lift(x_next_pred, u_next_true)[0]

    return preds

# ────────────────────────────────────────────────
# MAIN COMPARISON
# ────────────────────────────────────────────────
def compare():
    print("="*60)
    print("Unified Model Comparison")
    print("="*60)

    # 1. Load Metadata to get best Koopman params
    degree = 3
    lambda_reg = 1e-5
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            meta = json.load(f)
            degree = meta.get('degree', 3)
            lambda_reg = meta.get('lambda', 1e-5)
            print(f"Loaded Koopman params from metadata: d={degree}, lam={lambda_reg}")

    # 2. Load test data
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Run simulate_data.py.")
        return

    df = pd.read_csv(CSV_FILE)
    CA_true = df['CA_concentration'].values
    t = df['time'].values

    results = {}
    results['True'] = CA_true

    # 3. Koopman Prediction
    CA_koopman = predict_koopman(df, degree=degree, lambda_reg=lambda_reg)
    if CA_koopman is not None:
        results['Koopman'] = CA_koopman

    # 4. PINN / Vanilla
    try:
        if os.path.exists(PINN_MODEL):
            pinn_model = PINN_CSTR()
            pinn_model.load_state_dict(torch.load(PINN_MODEL))
            pinn_model.eval()
            CA_pinn, _ = rollout_prediction(pinn_model, df)
            results['PINN'] = CA_pinn
    except: pass

    try:
        if os.path.exists(VANILLA_MODEL):
            vanilla_model = Vanilla_NN()
            vanilla_model.load_state_dict(torch.load(VANILLA_MODEL))
            vanilla_model.eval()
            CA_vanilla, _ = rollout_prediction(vanilla_model, df)
            results['Vanilla'] = CA_vanilla
    except: pass

    # 5. Metrics & Plot
    plt.figure(figsize=(12, 6))
    plt.plot(t, CA_true, label='True', color='black', lw=2, alpha=0.7)

    colors = {'Koopman': 'red', 'PINN': 'orange', 'Vanilla': 'green'}

    print("\nPrediction Metrics (RMSE):")
    for name, data in results.items():
        if name == 'True': continue
        rmse = np.sqrt(np.mean((data - CA_true)**2))
        mae = np.mean(np.abs(data - CA_true))
        r2 = 1 - np.sum((data - CA_true)**2) / np.sum((CA_true - np.mean(CA_true))**2)

        print(f"{name}: RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.6f}")

        plt.plot(t, data, '--', label=f'{name} (RMSE={rmse:.4f})', color=colors.get(name, 'blue'))

    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('Open-Loop Prediction Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, 'prediction_comparison.png'))
    print(f"Saved plot to {SAVE_DIR}/prediction_comparison.png")

if __name__ == '__main__':
    compare()
