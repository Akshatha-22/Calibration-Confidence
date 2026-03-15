# Calibration-Failure-Detection
This is an 8th semester project.
# 📊 AI Confidence Calibration Failure Detection in Financial Time Series

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Dataset-FinSen-green.svg)](https://github.com/EagleAdelaide/FinSen_Dataset)

## 🎯 Overview

This repository contains the official implementation for **AI Confidence Calibration Failure Detection**, a systematic study of how five different neural architectures behave in terms of confidence calibration when applied to financial time series forecasting.

Well-calibrated models should be correct approximately 80% of the time when they report 80% confidence. However, modern deep learning models often exhibit **overconfidence** or **underconfidence**, leading to dangerous decisions in high-stakes applications like algorithmic trading and risk management.

## 🔍 What We Study

We compare **five architectures** on the same financial time series data:

| Model | Type | Why It's Included |
|-------|------|-------------------|
| **MLP** | (Sliding Window) | Feedforward | Baseline with no temporal memory |
| **Deep MLP** | Feedforward | Tests if depth affects calibration |
| **Vanilla RNN** | Recurrent | Simple recurrent baseline, clear gradient dynamics |
| **LSTM** | Gated Recurrent | Industry standard, masks gradient issues |
| **Residual MLP** | Hybrid | Skip connections for sequence modeling |

## 📈 Key Research Questions

1. **How does calibration error (ECE) evolve over time** for each architecture?
2. **Which models fail earliest** and which fail most predictably?
3. **Can gradient norms predict impending calibration failure**?
4. **Do different architectures have different "failure signatures"**?

## 💾 Dataset: FinSen

We use the **FinSen dataset**, which integrates:
- 📰 Economic/financial news from **197 countries** (2007-2023)
- 📊 S&P 500 stock market data
- 🧠 Sentiment scores from FinBERT model
- 📈 Volatility and price movement targets

> *Why FinSen?* The original FinSen paper specifically addresses model calibration using **Expected Calibration Error (ECE)** and achieves **3.34% ECE** with Focal Calibration Loss – making it the perfect benchmark for our study.

## 🏗️ Project Structure
├── data/
│ ├── finsen/ # FinSen dataset
│ └── preprocessing/ # Data loaders for all 5 models
├── models/
│ ├── mlp.py # MLP with sliding window
│ ├── deep_mlp.py # Deep MLP (4-5 layers)
│ ├── vanilla_rnn.py # Simple RNN
│ ├── lstm.py # LSTM with gates
│ └── residual_mlp.py # MLP with skip connections
├── calibration/
│ ├── ece.py # Expected Calibration Error
│ ├── reliability.py # Reliability diagrams
│ └── gradient_hooks.py # Gradient tracking
├── experiments/
│ ├── train.py # Unified training loop
│ ├── hyperparameter_tune.py # Grid search
│ └── robustness_tests.py # Noise, missing data
├── results/
│ ├── figures/ # ECE plots, gradient norms
│ └── tables/ # Comparative results
├── notebooks/
│ └── analysis.ipynb # Visualization and stats
├── requirements.txt
├── setup.py
└── README.md

text

 🚀 Getting Started
 Prerequisites

```bash
pip install -r requirements.txt


Quick Start

python
# Clone the FinSen dataset
git clone https://github.com/EagleAdelaide/FinSen_Dataset.git

# Run all 5 models with default settings
python experiments/train.py --dataset finsen --all_models

# Compute calibration metrics
python calibration/ece.py --results_dir ./results
📊 Key Metrics
Metric	Description
ECE	Expected Calibration Error = |confidence - accuracy|
Reliability Diagram	Visual plot of confidence vs. accuracy
Gradient Norm	‖∇W‖ tracks training stability
Failure Time	First timestep where ECE > threshold (e.g., 0.20)
Gradient-ECE Correlation	How well gradients predict calibration failure


📈 Expected Results
Based on our theoretical analysis, we expect:

Model	Failure Time	Predictability	Pattern
MLP	Never	N/A	Flat ECE
Deep MLP	Never	N/A	Slightly higher flat ECE
Vanilla RNN	Early (t≈25)	High (r>0.8)	Exponential rise
LSTM	Late (t≈75)	Low (r<0.4)	Sudden jump
Residual MLP	Variable	Medium	Chaotic


📝 Citation
If you use this code or the FinSen dataset for your research, please cite:

bibtex
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


👥 Team
Member	Role	Responsibilities
Member 1	Team Lead	Architecture, integration
Member 2	Model Implementer 1	MLP, Deep MLP, Residual MLP
Member 3	Model Implementer 2	RNN, LSTM
Member 4	Calibration Lead	ECE, visualization, analysis
Member 5	Experiment Runner	Training, tuning, results
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Eagle Adelaide Research for the FinSen dataset

Kaggle for hosting related competitions

Our academic advisors and reviewers

⭐ Star History
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
- Formatting: Black
- Linting: Flake8
- Type checking: mypy (optional)