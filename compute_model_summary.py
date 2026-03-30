import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Load the CSV file
df = pd.read_csv('results/hyperparameter_tuning/residual_mlp/results.csv')

# Filter rows where timestep is not null
df_filtered = df[df['timestep'].notna()].copy()

# Initialize results list
summary_data = []

# Group by model and compute metrics
for model in df_filtered['model'].unique():
    model_data = df_filtered[df_filtered['model'] == model].copy()
    
    # Compute failure time (first timestep where ECE > 0.15)
    failure_rows = model_data[model_data['ECE'] > 0.15].sort_values('timestep')
    failure_time = failure_rows['timestep'].iloc[0] if len(failure_rows) > 0 else None
    
    # Compute Spearman correlation between grad_norm and ECE
    # Handle NA values and numeric conversion
    grad_ece_data = model_data[['grad_norm', 'ECE']].copy()
    
    # Replace 'NA' string with NaN and convert to numeric
    grad_ece_data['grad_norm'] = pd.to_numeric(grad_ece_data['grad_norm'], errors='coerce')
    grad_ece_data = grad_ece_data.dropna()
    
    if len(grad_ece_data) > 1:
        grad_ece_corr, _ = spearmanr(grad_ece_data['grad_norm'], grad_ece_data['ECE'])
    else:
        grad_ece_corr = None
    
    # Compute mean ECE at specific timesteps
    ece_at_25 = model_data[model_data['timestep'] == 25]['ECE'].mean() if len(model_data[model_data['timestep'] == 25]) > 0 else None
    ece_at_50 = model_data[model_data['timestep'] == 50]['ECE'].mean() if len(model_data[model_data['timestep'] == 50]) > 0 else None
    ece_at_75 = model_data[model_data['timestep'] == 75]['ECE'].mean() if len(model_data[model_data['timestep'] == 75]) > 0 else None
    
    # Append to results
    summary_data.append({
        'model': model,
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
