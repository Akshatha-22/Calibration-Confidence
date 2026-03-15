"""
Data loader for FinSen dataset with multiple CSV files.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class FinSenDataset(Dataset):
    def __init__(self, data_path='data/finsen/raw', seq_length=50):
        """Load and merge FinSen CSV files into a sequence dataset.

        The dataset expects numeric features to be present (for regression).
        If the provided `data_path` has only text columns, it will attempt to fall
        back to a sibling `processed` folder containing numeric features.
        """
        self.seq_length = seq_length

        self.data = self._load_and_merge_csvs(data_path)
        print(f"Merged data: {self.data.shape[0]} rows, {self.data.shape[1]} columns")

        # Keep only numeric columns for model input.
        # Non-numeric columns like categorical labels or text cannot be converted directly.
        numeric_df = self.data.select_dtypes(include=[np.number])
        if numeric_df.shape[1] == 0:
            # Try to fall back to processed data if available
            if os.path.basename(os.path.normpath(data_path)) == 'raw':
                processed_path = os.path.join(os.path.dirname(data_path), 'processed')
                if os.path.isdir(processed_path):
                    print(f"No numeric columns found in raw CSVs; trying processed folder: {processed_path}")
                    self.data = self._load_and_merge_csvs(processed_path)
                    numeric_df = self.data.select_dtypes(include=[np.number])

        if numeric_df.shape[1] == 0:
            raise ValueError(
                "No numeric columns found in the merged dataset. "
                "Ensure the CSV files contain numeric features suitable for modeling."
            )

        self.values = numeric_df.values.astype(np.float32)
        if numeric_df.shape[1] != self.data.shape[1]:
            dropped = set(self.data.columns) - set(numeric_df.columns)
            print(f"Dropped non-numeric columns: {sorted(dropped)}")

    def _load_and_merge_csvs(self, data_path: str) -> pd.DataFrame:
        """Load all CSV files in a folder and merge them (on 'date' if present)."""
        all_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

        if len(all_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        print(f"Found {len(all_files)} CSV files: {all_files}")

        data_frames = []
        for file in all_files:
            file_path = os.path.join(data_path, file)
            df = pd.read_csv(file_path)
            print(f"Loaded {file}: {df.shape[0]} rows, {df.shape[1]} columns")
            data_frames.append(df)

        if len(data_frames) > 1:
            merged_df = data_frames[0]
            for df in data_frames[1:]:
                if 'date' in df.columns:
                    merged_df = pd.merge(merged_df, df, on='date', how='outer')
                else:
                    print(f"Warning: {file} has no 'date' column")
        else:
            merged_df = data_frames[0]

        if 'date' in merged_df.columns:
            merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
            merged_df = merged_df.sort_values('date').reset_index(drop=True)

        return merged_df
        
    def __len__(self):
        return len(self.values) - self.seq_length
    
    def __getitem__(self, idx):
        """
        Returns:
            sequence: (seq_length, num_features)
            target: next value(s) to predict
        """
        sequence = self.values[idx:idx + self.seq_length]
        target = self.values[idx + self.seq_length]
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)
    
    def get_info(self):
        """Return dataset information"""
        return {
            'total_rows': len(self.values),
            'sequences': len(self),
            'features': self.values.shape[1],
            'date_range': [self.data['date'].min(), self.data['date'].max()] if 'date' in self.data.columns else 'Unknown'
        }