import numpy as np
import pandas as pd
import os

results_map = {
    'vanilla_rnn': 'd:/Calibration-Confidence/results/rnn_fixed.npz',
    'lstm': 'd:/Calibration-Confidence/results/lstm_fixed.npz',
    'residual': 'd:/Calibration-Confidence/results/residual_fixed.npz'
}

for model_name, npz_path in results_map.items():
    if not os.path.exists(npz_path):
        print(f"Skipping {model_name}, {npz_path} not found")
        continue
        
    data = np.load(npz_path, allow_pickle=True)
    targets = data['targets']
    preds = data['predictions']
    
    # Per-feature variance check
    vars = np.var(targets, axis=0)
    mse = np.mean((preds - targets)**2)
    # Global variance across all flattened features
    global_var = np.mean(vars)
    
    rmse = np.sqrt(mse)
    r2 = 1 - (mse / global_var)
    # Calibration error was saved as an array during collect_results
    ce = data.get('calibration_error')
    if ce is not None:
        if isinstance(ce, (np.ndarray, list)) and len(ce) > 0:
            ce = ce[0]
        else:
            ce = float(ce)
    else:
        ce = 0.0
        
    print(f"Model: {model_name}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R2: {r2:.6f}")
    print(f"  CE: {ce:.6f}")
    
    # Update the corresponding results.csv
    csv_path = f'd:/Calibration-Confidence/results/hyperparameter_tuning/{model_name}/results.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Create a new row based on the last one
        new_row = df.iloc[-1].copy()
        new_row['timestep'] = 999  # Mark as final fixed run or very high epoch
        new_row['RMSE'] = rmse
        new_row['R2'] = r2
        new_row['ECE'] = ce
        
        # Add to dataframe
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"  Updated {csv_path}")
    else:
        print(f"  Warning: {csv_path} not found, could not update master results.")
