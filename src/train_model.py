
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Locate the project folders
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_FOLDER / "data" / "emails.csv"
MODEL_FOLDER = PROJECT_FOLDER / "models"
MODEL_PATH = MODEL_FOLDER / "spam_model.pkl"


def train_model():
    # Load the email dataset
    df = pd.read_csv(DATA_PATH)

    # Keep only rows with valid text and labels
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"] != ""]

    # Convert labels into consistent values
    valid_labels = {"ham", "spam"}
    df = df[df["label"].isin(valid_labels)]

    if df["label"].nunique() != 2 or df["label"].value_counts().min() < 2:
        raise ValueError("Need at least two examples of each label.")

    # Separate email messages from their known labels
    X = df["text"]
    y = df["label"]

    # Split the data into training and testing portions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Build a pipeline: text -> TF-IDF features -> classifier
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ])

    # Train the model on the training messages
    model.fit(X_train, y_train)

    # Evaluate it on messages held aside for testing
    predictions = model.predict(X_test)

    print("\n--- Model Evaluation ---")
    print(f"Test messages: {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # Save the trained pipeline
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()