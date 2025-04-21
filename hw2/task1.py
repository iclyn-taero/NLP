# task1_basic_stats.py

import pandas as pd

def run_task1(train_df):
    print("\n🔹 TASK 1: Basic Dataset Statistics")

    print(f"Total samples: {len(train_df)}")

    word_counts = train_df['content'].apply(lambda x: len(str(x).split()))
    avg_words = word_counts.mean()
    max_words = word_counts.max()
    min_words = word_counts.min()

    print(f"Average words per sample: {avg_words:.2f}")
    print(f"Max words in a sample: {max_words}")
    print(f"Min words in a sample: {min_words}")

    if 'label' in train_df.columns:
        print("\nLabel Distribution:")
        print(train_df['label'].value_counts())

if __name__ == "__main__":
    try:
        df = pd.read_csv("data/train.csv")  # Replace with your actual path
        run_task1(df)
    except Exception as e:
        print("❌ Error loading data:", e)
