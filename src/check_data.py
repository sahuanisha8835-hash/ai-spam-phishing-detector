
import pandas as pd
from pathlib import Path

# Find the main project folder
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# Locate the email dataset
DATA_PATH = PROJECT_FOLDER / "data" / "emails.csv"


def check_dataset():
    # Load the CSV file
    df = pd.read_csv(DATA_PATH)

    print("\n--- Dataset Preview ---")
    print(df.head())

    print("\n--- Dataset Information ---")
    print(f"Total emails: {len(df)}")
    print(f"Missing values:\n{df.isnull().sum()}")

    print("\n--- Label Distribution ---")
    print(df["label"].value_counts())

    # Check required columns
    required_columns = {"text", "label"}

    if required_columns.issubset(df.columns):
        print("\nDataset structure is correct.")
    else:
        print("\nError: Required columns are missing.")


if __name__ == "__main__":
    check_dataset()