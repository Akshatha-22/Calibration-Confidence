import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def explore_finsen_dataset(data_path='data/finsen/raw/FinSen_US_Categorized.csv'):
    """
    Load and explore the FinSen dataset.
    Generates plots and prints statistics.
    """
    print("--- Exploring FinSen Dataset ---")

    # Load the categorized dataset
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nCategory distribution:")
    category_counts = df['Category'].value_counts()
    print(category_counts)

    # Plot category distribution
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind='bar')
    plt.title('Distribution of Categories in FinSen Dataset')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/figures/category_distribution.png')
    plt.show()

    print("\nTag distribution (top 10):")
    tag_counts = df['Tag'].value_counts().head(10)
    print(tag_counts)

    # Plot tag distribution
    plt.figure(figsize=(10, 6))
    tag_counts.plot(kind='bar')
    plt.title('Top 10 Tags in FinSen Dataset')
    plt.xlabel('Tag')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/figures/tag_distribution.png')
    plt.show()

    print("\nContent length statistics:")
    df['content_length'] = df['Content'].str.len()
    print(df['content_length'].describe())

    # Plot content length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['content_length'], bins=50, alpha=0.7)
    plt.title('Distribution of Content Lengths')
    plt.xlabel('Content Length (characters)')
    plt.ylabel('Frequency')
    plt.savefig('results/figures/content_length_distribution.png')
    plt.show()

    print("\nExploration complete! Check results/figures/ for plots.")

    return df

if __name__ == "__main__":
    explore_finsen_dataset()