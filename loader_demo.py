# test_loader.py
from preprocessing.finsen_loader import FinSenDataset

# Load dataset
dataset = FinSenDataset(data_path='data/finsen/raw', seq_length=50)

# Print info
info = dataset.get_info()
print(f"Dataset info: {info}")

# Test getting one sample
seq, target = dataset[0]
print(f"Sequence shape: {seq.shape}")
print(f"Target shape: {target.shape}")

print("✅ Loader working!")