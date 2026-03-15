import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")
print(f"Installation successful! ✅")

# To explore the dataset, run:
# from data.preprocessing.dataset_explorer import explore_finsen_dataset
# explore_finsen_dataset()
    
import numpy as np

data = np.load("results/first_results.npz")
keys = list(data.keys())
print("Available keys:", keys)

predictions = data["predictions"]
targets = data["targets"]
print(f"Predictions shape: {predictions.shape}")
print(f"Targets shape: {targets.shape}")

if "per_example_loss" in data:
    losses = data["per_example_loss"]
    print(f"Per-example loss shape: {losses.shape}, mean: {float(losses.mean()):.4f}")

# Optional: only present when model outputs multi-class logits (classification)
if "confidences" in data:
    confidences = data["confidences"]
    print(f"Confidences shape: {confidences.shape}, mean: {float(confidences.mean()):.4f}")
if "ece" in data:
    ece = data["ece"]
    print(f"ECE: {float(ece[0]):.4f}")
if "reliability" in data:
    print("Reliability:", data["reliability"])
if "gradient_norms" in data:
    print(f"Gradient norms shape: {data['gradient_norms'].shape}")








