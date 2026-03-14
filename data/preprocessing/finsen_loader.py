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
        """
        Load and merge multiple FinSen CSV files
        
        Args:
            data_path: Path to folder containing CSV files
            seq_length: Sequence length for models
        """
        self.seq_length = seq_length
        
        # Load all CSV files from the raw folder
        all_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        
        if len(all_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        
        print(f"Found {len(all_files)} CSV files: {all_files}")
        
        # Load and merge all CSV files
        data_frames = []
        for file in all_files:
            file_path = os.path.join(data_path, file)
            df = pd.read_csv(file_path)
            print(f"Loaded {file}: {df.shape[0]} rows, {df.shape[1]} columns")
            data_frames.append(df)
        
        # Merge based on date column (assuming all have 'date')
        if len(data_frames) > 1:
            # Start with first file
            merged_df = data_frames[0]
            # Merge with others on date
            for df in data_frames[1:]:
                if 'date' in df.columns:
                    merged_df = pd.merge(merged_df, df, on='date', how='outer')
                else:
                    print(f"Warning: {file} has no 'date' column")
        else:
            merged_df = data_frames[0]
        
        # Sort by date and reset index
        if 'date' in merged_df.columns:
            merged_df['date'] = pd.to_datetime(merged_df['date'])
            merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        self.data = merged_df
        print(f"Merged data: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Convert to numpy for faster access (exclude date column if present)
        if 'date' in self.data.columns:
            self.values = self.data.drop('date', axis=1).values.astype(np.float32)
        else:
            self.values = self.data.values.astype(np.float32)
        
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