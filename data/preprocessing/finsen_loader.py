"""
Data loader for FinSen dataset with multiple CSV files.
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset


class FinSenDataset(Dataset):
    def __init__(
        self,
        data_path: str = 'data/finsen/raw',
        seq_length: int = 50,
        text_vectorizer_max_features: int = 128,
    ):
        """Load and merge FinSen CSV files into a sequence dataset.

        For regression, vectorizes text to TF-IDF and combines with any numeric features.
        For adequate feature dimension, always includes vectorized text features.
        """
        self.seq_length = seq_length
        self.text_vectorizer_max_features = text_vectorizer_max_features

        self.data = self._load_and_merge_csvs(data_path)
        print(f"Merged data: {self.data.shape[0]} rows, {self.data.shape[1]} columns")

        numeric_df = self.data.select_dtypes(include=[np.number])
        
        # Always vectorize text to have enough features for regression
        text_df = self._vectorize_text_columns(self.data)
        
        # Combine numeric and text features
        if text_df is not None and len(text_df) > 0:
            if len(numeric_df) > 0:
                # Add content_length or other numeric features
                numeric_df_subset = numeric_df[[col for col in numeric_df.columns if col not in ['Title', 'Tag', 'Content', 'Category']]]
                if len(numeric_df_subset.columns) > 0:
                    combined_df = pd.concat([text_df, numeric_df_subset], axis=1)
                else:
                    combined_df = text_df
            else:
                combined_df = text_df
        elif len(numeric_df) > 0:
            combined_df = numeric_df
        else:
            raise ValueError(
                "Could not create any numeric features from the dataset. "
                "Ensure the CSV files contain numeric features or text that can be vectorized."
            )

        self.values = combined_df.astype(np.float32).values
        print(f"Final feature dimension: {self.values.shape[1]}")

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

    def _vectorize_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert available text columns into TF-IDF features."""
        text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
        if len(text_columns) == 0:
            return pd.DataFrame()

        print(f"Vectorizing text columns for numeric features: {text_columns}")
        text_data = (
            df[text_columns]
            .fillna('')
            .astype(str)
            .agg(' '.join, axis=1)
        )
        vectorizer = TfidfVectorizer(
            max_features=self.text_vectorizer_max_features,
            stop_words='english',
        )
        features = vectorizer.fit_transform(text_data).toarray()
        feature_names = [f'text_feat_{i}' for i in range(features.shape[1])]
        vectorized_df = pd.DataFrame(features, columns=feature_names)
        print(f"TF-IDF matrix shape: {vectorized_df.shape}")
        return vectorized_df

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
            'date_range': (
                [self.data['date'].min(), self.data['date'].max()]
                if 'date' in self.data.columns
                else 'Unknown'
            ),
        }
