# 📊 AI Confidence Calibration Failure Detection

**8th Semester Project** | Analyzing confidence calibration in neural networks for financial forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What is This Project?

This project studies **how well neural networks are calibrated** when predicting financial time series. A well-calibrated model is correct 80% of the time when it says 80% confidence. But most modern models are overconfident, which is dangerous in finance.

We test **5 different neural network architectures** on the same data and measure which ones fail first and which have the most stable confidence.

## 🔬 The 5 Models We Compare

| Model | Type | Purpose |
|-------|------|---------|
| **MLP** | Feedforward | Simple baseline (no memory) |
| **Deep MLP** | Feedforward | Test if depth affects calibration |
| **Vanilla RNN** | Recurrent | Simple recurrence with clear gradient flow |
| **LSTM** | Gated Recurrent | Industry standard with gates |
| **Residual MLP** | Hybrid | Feedforward + skip connections |

## 📊 Key Metrics

- **ECE (Expected Calibration Error)**: Gap between predicted confidence and actual correctness
- **Brier Score**: Overall prediction quality (lower is better)
- **Confidence Stats**: Average confidence on correct vs incorrect predictions

## 💾 Dataset: FinSen

Uses the **FinSen dataset** covering:
- Financial news from 197 countries (2007-2023)
- S&P 500 stock prices
- Sentiment analysis via FinBERT
- Price movement targets

## 📂 Project Structure

```
./
├── calibration/
│   ├── ece.py                          # Expected Calibration Error computation
│   ├── confidence_tracking.py           # Confidence tracking utilities
│   ├── gradient_hooks.py                # Gradient monitoring hooks
│   └── reliability.py                   # Reliability diagram generation
├── checkpoints_deep_mlp/                # Pre-trained Deep MLP weights
├── checkpoints_mlp/                     # Pre-trained MLP weights
├── checkpoints_vanilla_rnn/             # Pre-trained RNN weights
├── data/finsen/
│   ├── processed/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── raw/
│       └── FinSen_US.csv
├── experiments/
│   ├── train.py                         # Main training script
│   ├── hyperparameter_tune.py           # Hyperparameter tuning script
│   ├── plot_ece_over_time.py            # ECE evolution plots
│   ├── plot_sensitivity.py              # Sensitivity analysis
│   ├── robustness_tests.py              # Robustness testing
│   └── inspect_results.py               # Results inspection utilities
├── models/
│   ├── mlp.py                           # MLP architecture
│   ├── deep_mlp.py                      # Deep MLP architecture
│   ├── vanilla_rnn.py                   # Vanilla RNN architecture
│   ├── lstm.py                          # LSTM architecture
│   └── residual_mlp.py                  # Residual MLP architecture
├── preprocessing/
│   ├── finsen_loader.py                 # FinSen dataset loader
│   ├── mlp_loader.py                    # MLP data loader
│   ├── rnn_loader.py                    # RNN data loader
│   └── data_loaders.py                  # Generic data loaders
├── results/
│   ├── figures/
│   │   ├── all_models_loss.png
│   │   ├── ece_over_time.png
│   │   ├── gradient_norms.png
│   │   ├── model_performance_comparison.png
│   │   └── ... (more plots)
│   ├── sensitivity_plots/
│   │   ├── mlp.png
│   │   ├── deep_mlp.png
│   │   ├── lstm.png
│   │   ├── vanilla_rnn.png
│   │   └── residual_mlp.png
│   └── hyperparameter_tuning/
│       ├── mlp/
│       ├── lstm/
│       ├── deep_mlp/
│       └── ... (tuning results)
├── notebooks/
│   ├── model_tuning.ipynb
│   ├── results_report.ipynb
│   └── colab_train_and_report.ipynb
├── models.py                            # Model utilities
├── model_summary.csv                    # Summary of model metrics
├── ece_quality_report.csv               # ECE quality report
├── requirements.txt
├── setup.py
└── TRAINING_PIPELINE_FIXES.md
```
## 🛠️ Main Components

- **calibration/** - ECE computation, gradient hooks, reliability diagrams
- **models/** - 5 neural network architectures for comparison
- **experiments/train.py** - Main training script with all fixes applied
- **data/finsen/** - FinSen dataset with train/val/test splits
- **results/** - Output directory for trained models and metrics

## 🚀 Getting Started
### Prerequisites

```bash
pip install -r requirements.txt
```

### Quick Start

```bash
# Clone the FinSen dataset
git clone https://github.com/EagleAdelaide/FinSen_Dataset.git

# Run all 5 models with default settings
python experiments/train.py --dataset finsen --all_models

# Compute calibration metrics
python calibration/ece.py --results_dir ./results
```

## 📊 Key Metrics
| Metric | Description |
|--------|-------------|
| ECE | Expected Calibration Error = \|confidence - accuracy\| |
| Reliability Diagram | Visual plot of confidence vs. accuracy |
| Gradient Norm | ‖∇W‖ tracks training stability |
| Failure Time | First timestep where ECE > threshold (e.g., 0.20) |
| Gradient-ECE Correlation | How well gradients predict calibration failure |


## 📈 Expected Results
Based on our theoretical analysis, we expect:

| Model | Failure Time | Predictability | Pattern |
|-------|--------------|----------------|---------|
| MLP | Never | N/A | Flat ECE |
| Deep MLP | Never | N/A | Slightly higher flat ECE |
| Vanilla RNN | Early (t≈25) | High (r>0.8) | Exponential rise |
| LSTM | Late (t≈75) | Low (r<0.4) | Sudden jump |
| Residual MLP | Variable | Medium | Chaotic |


## 📝 Citation
If you use this code or the FinSen dataset for your research, please cite:

```bibtex
@misc{calibration2025,
  author = {Your Team Name},
  title = {AI Confidence Calibration Failure Detection in Financial Time Series},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/your-repo}
}

@article{finsen2024,
  title={FinSen: A Dataset for Financial Sentiment Analysis with Calibration-Aware Learning},
  author={Eagle Adelaide Research},
  year={2024},
  url={https://github.com/EagleAdelaide/FinSen_Dataset}
}
```

## 👥 Team
| Member | Role | Responsibilities |
|--------|------|------------------|
| Member 1 | Team Lead | Architecture, integration |
| Member 2 | Model Implementer 1 | MLP, Deep MLP, Residual MLP |
| Member 3 | Model Implementer 2 | RNN, LSTM |
| Member 4 | Calibration Lead | ECE, visualization, analysis |
| Member 5 | Experiment Runner | Training, tuning, results |

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- Eagle Adelaide Research for the FinSen dataset

- Kaggle for hosting related competitions

- Our academic advisors and reviewers

## ⭐ Star History
If you find this project useful, please consider giving it a star! It helps others discover our work.

# Project Coding Standards

## 1. Python Version
- Use Python 3.11+

## 2. Naming
- Variables: snake_case (e.g., `hidden_size`)
- Functions: snake_case (e.g., `train_model()`)
- Classes: PascalCase (e.g., `VanillaRNN`)
- Constants: UPPER_CASE (e.g., `MAX_EPOCHS = 100`)

## 3. Imports (in this order)
1. Standard library (os, sys, json)
2. Third-party (torch, numpy, matplotlib)
3. Local modules (models.rnn, calibration.ece)

## 4. Line Length
- Maximum 88 characters (Black default)

## 5. Docstrings
- Every function needs a docstring explaining:
  - What it does
  - Parameters
  - Returns

## 6. Comments
- Explain WHY, not WHAT (the code shows what)
- No commented-out code

## 7. Git Commits
- Use present tense: "Add ECE computation" not "Added ECE computation"
- First line <50 chars, then blank line, then details

## 8. Tools We Use
- Core libraries (from requirements.txt): PyTorch, NumPy, pandas, Matplotlib, scikit-learn, SciPy
- Notebooks: Jupyter, Google Colab
- Hyperparameter tuning: Optuna
- Optional experiment logging: TensorBoard, Weights & Biases (wandb)
