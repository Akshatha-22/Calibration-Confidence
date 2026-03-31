import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import traceback

# Add root to sys.path
sys.path.append(os.getcwd())

from experiments.train import build_mlp, build_deep_mlp, build_vanilla_rnn, build_lstm, build_residual_mlp
from data.preprocessing.data_loaders import get_mlp_loaders, get_rnn_loaders
from calibration.ece import regression_calibration_error

def _compute_rmse_r2(preds, targets):
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    rmse = np.sqrt(np.mean((preds - targets)**2))
    ss_res = np.sum((targets - preds)**2)
    ss_tot = np.sum((targets - np.mean(targets))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(rmse), float(r2)

def evaluate_model_at_horizon(model_name, checkpoint_path, horizons=[25, 50, 75]):
    print(f"\nProcessing {model_name} from {checkpoint_path}...")
    device = torch.device("cpu")
    
    if not os.path.exists(checkpoint_path):
        print(f"  Missing checkpoint: {checkpoint_path}")
        return []

    results_rows = []
    for h in horizons:
        try:
            print(f"  Evaluating at t={h}...")
            # Create loaders for this horizon
            if model_name in ('rnn', 'lstm'):
                _, _, test_loader = get_rnn_loaders(
                    train_path="data/finsen/processed/train.csv",
                    val_path="data/finsen/processed/val.csv",
                    test_path="data/finsen/processed/test.csv",
                    batch_size=128,
                    max_seq_len=h
                )
            else:
                _, _, test_loader = get_mlp_loaders(
                    train_path="data/finsen/processed/train.csv",
                    val_path="data/finsen/processed/val.csv",
                    test_path="data/finsen/processed/test.csv",
                    batch_size=128,
                    max_features=2000
                )
            
            # Get num_features/output_size
            x_sample, y_sample = next(iter(test_loader))
            if model_name in ('rnn', 'lstm'):
                num_features = 1
            else:
                num_features = x_sample.shape[-1]
            output_size = 12 
            
            # Build model
            if model_name == 'mlp':
                model = build_mlp(seq_len=1, num_features=num_features, hidden_sizes=(128, 64), output_size=output_size)
            elif model_name == 'deep':
                model = build_deep_mlp(seq_len=1, num_features=num_features, hidden_sizes=(256, 128, 64), output_size=output_size)
            elif model_name == 'rnn':
                model = build_vanilla_rnn(seq_len=h, num_features=num_features, hidden_size=64, output_size=output_size)
            elif model_name == 'lstm':
                model = build_lstm(seq_len=h, num_features=num_features, hidden_size=64, output_size=output_size)
            elif model_name == 'residual':
                model = build_residual_mlp(seq_len=1, num_features=num_features, hidden_size=128, num_blocks=3, output_size=output_size)
            
            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(device)
            model.eval()
            
            # Run evaluation
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    if model_name in ('rnn', 'lstm'):
                        pass
                    else:
                        x = x.view(x.size(0), -1)
                    
                    logits = model(x)
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    all_preds.append(preds)
                    all_targets.append(y.cpu().numpy())
            
            p = np.concatenate(all_preds)
            t = np.concatenate(all_targets)
            
            rmse, r2 = _compute_rmse_r2(p, t)
            ce = regression_calibration_error(p, t)
            
            print(f"    t={h}: RMSE={rmse:.4f}, R2={r2:.4f}, CE={ce:.4f}")
            results_rows.append({
                'trial_id': 1, 'model': model_name, 'ECE': ce, 'RMSE': rmse, 'R2': r2,
                'grad_norm': checkpoint.get('history', {}).get('train_grad_norm', [0.1])[-1],
                'timestep': h, 'task': 'classification'
            })
        except Exception as e:
            print(f"  Error at t={h}: {e}")
            traceback.print_exc()
            
    return results_rows

def main():
    models = {
        'mlp': 'results/checkpoints/mlp.pt',
        'deep_mlp': 'results/checkpoints/deep.pt',
        'vanilla_rnn': 'results/checkpoints/rnn.pt',
        'lstm': 'results/checkpoints/lstm.pt',
        'residual_mlp': 'results/checkpoints/residual.pt'
    }
    
    for m_name, ckpt in models.items():
        short_name = m_name.replace('_mlp', '').replace('vanilla_', '')
        rows = evaluate_model_at_horizon(short_name, ckpt)
        if rows:
            df = pd.DataFrame(rows)
            target_dir = f"results/hyperparameter_tuning/{m_name}"
            os.makedirs(target_dir, exist_ok=True)
            path = f"{target_dir}/results.csv"
            df.to_csv(path, index=False)
            print(f"  SUCCESS: Saved to {path}")
        else:
            print(f"  FAILED: No rows collected for {m_name}")

if __name__ == "__main__":
    main()
