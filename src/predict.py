
import joblib
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_FOLDER / "models" / "spam_model.pkl"


def predict_email():
    if not MODEL_PATH.exists():
        print("Model not found. Run: python src\\train_model.py")
        return

    model = joblib.load(MODEL_PATH)

    email_text = input("\nEnter an email message: ").strip()

    if not email_text:
        print("Please enter an email message.")
        return

    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0]
    classes = model.classes_

    spam_index = list(classes).index("spam")
    spam_probability = probabilities[spam_index] * 100

    print("\n--- Prediction Result ---")

    if prediction == "spam":
        print("Classification: SPAM")
    else:
        print("Classification: NOT SPAM")

    print(f"Model's estimated spam probability: {spam_probability:.1f}%")
    print("Note: This is an experimental model, not a reliable security verdict.")


if __name__ == "__main__":
    predict_email()