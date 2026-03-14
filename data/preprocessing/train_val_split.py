import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42, stratify_col='Category'):
    """
    Split the dataset into training, validation, and test sets.

    Parameters:
    - df: pandas DataFrame
    - train_size: float, proportion for training (default 0.7)
    - val_size: float, proportion for validation (default 0.15)
    - test_size: float, proportion for test (default 0.15)
    - random_state: int, for reproducibility
    - stratify_col: str, column to stratify on (default 'Category')

    Returns:
    - train_df, val_df, test_df: pandas DataFrames
    """
    assert train_size + val_size + test_size == 1.0, "Proportions must sum to 1.0"

    # First, split into train and temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=df[stratify_col] if stratify_col in df.columns else None
    )

    # Then split temp into val and test
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size / (val_size + test_size),
        random_state=random_state,
        stratify=temp_df[stratify_col] if stratify_col in temp_df.columns else None
    )

    print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df

if __name__ == "__main__":
    # Example usage
    from dataset_explorer import explore_finsen_dataset

    df = explore_finsen_dataset()
    train_df, val_df, test_df = split_dataset(df)

    # Optionally save the splits
    train_df.to_csv('data/finsen/processed/train.csv', index=False)
    val_df.to_csv('data/finsen/processed/val.csv', index=False)
    test_df.to_csv('data/finsen/processed/test.csv', index=False)

    print("Splits saved to data/finsen/processed/")