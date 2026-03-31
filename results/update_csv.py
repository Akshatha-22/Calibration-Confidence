import os
import sys
import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calibration.ece import regression_calibration_error

results_map = {
    'vanilla_rnn': 'd:/Calibration-Confidence/results/rnn_fixed.npz',
    'lstm': 'd:/Calibration-Confidence/results/lstm_fixed.npz',
    'residual_mlp': 'd:/Calibration-Confidence/results/residual_fixed.npz',
    'mlp': 'd:/Calibration-Confidence/results/mlp_fixed.npz',
    'deep_mlp': 'd:/Calibration-Confidence/results/deep_fixed.npz'
}

for model_name, npz_path in results_map.items():
    if not os.path.exists(npz_path):
        continue
        
    data = np.load(npz_path, allow_pickle=True)
    t = data['targets'].astype(np.float64)
    if 'pred_labels' in data:
        # Classification: use predicted labels for RMSE
        p_arr = data['pred_labels'].astype(np.float64).reshape(t.shape)
    else:
        # Regression: use raw predictions
        p_arr = data['predictions'].astype(np.float64)
    
    # Ensure same shape for MSE
    if p_arr.shape != t.shape and p_arr.ndim == 2 and t.ndim == 1:
        t = t.reshape(-1, 1).repeat(p_arr.shape[1], axis=1)
    
    mse = np.mean((p_arr - t)**2)
    var = np.var(t)
    
    rmse = np.sqrt(mse)
    r2 = 1 - (mse / var) if var > 0 else 0.0
    
    ce = regression_calibration_error(p_arr, t, n_bins=10)
    
    print(f"{model_name}: RMSE={rmse:.6f}, R2={r2:.6f}, CE={ce:.6f}")
    
    csv_path = f'd:/Calibration-Confidence/results/hyperparameter_tuning/{model_name}/results.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        new_row = df.iloc[-1].copy()
        new_row['timestep'] = 1000
        new_row['RMSE'] = rmse
        new_row['R2'] = r2
        new_row['ECE'] = ce
        new_row['task'] = 'regression'
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
