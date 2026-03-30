import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

# Initialize results list
summary_data = []

# Model directories to process
model_dirs = ['deep_mlp', 'lstm', 'mlp', 'residual_mlp', 'vanilla_rnn']

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
    
    # Compute failure time (first timestep where ECE > 0.15)
    failure_rows = df_filtered[df_filtered['ECE'] > 0.15].sort_values('timestep')
    failure_time = failure_rows['timestep'].iloc[0] if len(failure_rows) > 0 else None
    
    # Compute Spearman correlation between grad_norm and ECE
    # Handle NA values and numeric conversion
    grad_ece_data = df_filtered[['grad_norm', 'ECE']].copy()
    
    # Replace 'NA' string with NaN and convert to numeric
    grad_ece_data['grad_norm'] = pd.to_numeric(grad_ece_data['grad_norm'], errors='coerce')
    grad_ece_data = grad_ece_data.dropna()
    
    if len(grad_ece_data) > 1:
        grad_ece_corr, _ = spearmanr(grad_ece_data['grad_norm'], grad_ece_data['ECE'])
    else:
        grad_ece_corr = None
    
    # Compute mean ECE at specific timesteps
    ece_at_25 = df_filtered[df_filtered['timestep'] == 25]['ECE'].mean() if len(df_filtered[df_filtered['timestep'] == 25]) > 0 else None
    ece_at_50 = df_filtered[df_filtered['timestep'] == 50]['ECE'].mean() if len(df_filtered[df_filtered['timestep'] == 50]) > 0 else None
    ece_at_75 = df_filtered[df_filtered['timestep'] == 75]['ECE'].mean() if len(df_filtered[df_filtered['timestep'] == 75]) > 0 else None
    
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

print(summary_df)
