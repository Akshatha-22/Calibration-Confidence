"""
Compute model-level summary metrics from hyperparameter tuning results.

Metrics computed per model:
- ECE@25, ECE@50, ECE@75: Expected Calibration Error at 25%, 50%, 75% of max training time
- failure_time: First timestep where ECE exceeds (mean + 1*std) threshold
- grad_ece_corr: Spearman correlation between gradient norm and ECE
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

# Initialize results list
summary_data = []

# Model directories to process
model_dirs = ['deep_mlp', 'lstm', 'mlp', 'residual_mlp', 'vanilla_rnn']

# Helper function to get ECE at closest relative timestep
def get_ece_at_percentile(group, percent):
    """Get mean ECE at the closest timestep to a relative position (0-1)."""
    max_t = group['timestep'].max()
    target_t = percent * max_t
    
    available_ts = sorted(group['timestep'].unique())
    if len(available_ts) == 0:
        return None
    
    # Find closest timestep to target
    closest_ts = min(available_ts, key=lambda x: abs(x - target_t))
    matching = group[group['timestep'] == closest_ts]
    return matching['ECE'].mean() if len(matching) > 0 else None

# Process each model directory
for model_name in model_dirs:
    csv_path = f'results/hyperparameter_tuning/{model_name}/results.csv'
    
    # Check if file exists
    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found, skipping {model_name}")
        continue
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter rows where timestep is not null
    df_filtered = df[df['timestep'].notna()].copy()
    
    if len(df_filtered) == 0:
        print(f"Warning: No rows with non-null timestep in {model_name}, skipping")
        continue
    
    # Compute ECE at relative timesteps (closest available)
    ece_at_25 = get_ece_at_percentile(df_filtered, 0.25)
    ece_at_50 = get_ece_at_percentile(df_filtered, 0.50)
    ece_at_75 = get_ece_at_percentile(df_filtered, 0.75)
    
    # Compute dynamic failure threshold: mean + 1 std
    # (Default failure threshold if insufficient data)
    ece_mean = df_filtered['ECE'].mean()
    ece_std = df_filtered['ECE'].std()
    failure_threshold = ece_mean + ece_std
    
    # Compute failure time (first timestep where ECE exceeds threshold)
    failure_rows = df_filtered[df_filtered['ECE'] > failure_threshold].sort_values('timestep')
    failure_time = failure_rows['timestep'].iloc[0] if len(failure_rows) > 0 else None
    
    # Compute Spearman correlation between grad_norm and ECE
    grad_ece_data = df_filtered[['grad_norm', 'ECE']].copy()
    grad_ece_data['grad_norm'] = pd.to_numeric(grad_ece_data['grad_norm'], errors='coerce')
    grad_ece_data = grad_ece_data.dropna()
    
    # Only compute correlation if we have at least 3 valid pairs
    if len(grad_ece_data) > 2:
        grad_ece_corr, _ = spearmanr(grad_ece_data['grad_norm'], grad_ece_data['ECE'])
    else:
        grad_ece_corr = None
    
    # Append to results
    summary_data.append({
        'model': model_name,
        'ECE@25': ece_at_25,
        'ECE@50': ece_at_50,
        'ECE@75': ece_at_75,
        'failure_time': failure_time,
        'grad_ece_corr': grad_ece_corr
    })

# Create summary DataFrame
summary_df = pd.DataFrame(summary_data)

# Save to CSV
summary_df.to_csv('model_summary.csv', index=False)

print("\n" + "="*70)
print("MODEL SUMMARY METRICS")
print("="*70)
print(summary_df.to_string(index=False))
print("="*70)
