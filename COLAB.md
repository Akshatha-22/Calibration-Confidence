## Colab Quickstart

This guide gets you from a fresh Colab runtime to a tuning run in one cell.

**One-cell install + run (paste into a single Colab cell):**
```bash
# If you opened this notebook from a different repo, clone this one first:
# !git clone <YOUR_REPO_URL>
# %cd Calibration-Confidence

!pip -q install -r requirements.txt

# If you ALREADY have the dataset, copy it into the expected path:
!mkdir -p data/finsen
!cp -r /content/<YOUR_DATASET_ROOT>/raw data/finsen/

# Optional: download the FinSen dataset if you don't have it
# !git clone https://github.com/EagleAdelaide/FinSen_Dataset.git
# !mkdir -p data/finsen
# !cp -r FinSen_Dataset/data/raw data/finsen/

# Run a grid search (example: MLP)
!python experiments/hyperparameter_tune.py --model mlp --method grid
```

## Dataset Notes (FinSen)
- Source: GitHub repo `EagleAdelaide/FinSen_Dataset`
- Training expects raw CSVs under `data/finsen/raw`
- If you already have the data in Google Drive, mount Drive and copy to that path:
```bash
from google.colab import drive
drive.mount("/content/drive")

# Example: copy from Drive into the expected path
!mkdir -p data/finsen
!cp -r /content/drive/MyDrive/FinSen/raw data/finsen/
```

## Common Runs
```bash
# Deep MLP grid
!python experiments/hyperparameter_tune.py --model deep --method grid

# RNN grid
!python experiments/hyperparameter_tune.py --model rnn --method grid

# LSTM grid
!python experiments/hyperparameter_tune.py --model lstm --method grid

# Residual MLP grid
!python experiments/hyperparameter_tune.py --model residual --method grid
```
