
import streamlit as st
import joblib
from pathlib import Path


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Spam & Phishing Detector",
    page_icon="🛡️",
    layout="centered"
)


# -----------------------------
# Find and load the model
# -----------------------------
PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_FOLDER / "models" / "spam_model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# -----------------------------
# App interface
# -----------------------------
st.title("🛡️ AI Spam & Phishing Detector")

st.write(
    "Analyze an email message using a machine-learning "
    "model trained to classify spam and legitimate messages."
)

st.info(
    "Learning project: this model uses a very small sample "
    "dataset. Do not rely on it to decide whether a real email "
    "or link is safe."
)

if model is None:
    st.error("Trained model not found.")

    st.code("python src\\train_model.py")

    st.stop()


email_text = st.text_area(
    "Paste your email message",
    height=180,
    placeholder="Enter the email text here..."
)

if st.button("Analyze Email", type="primary", use_container_width=True):

    if not email_text.strip():
        st.warning("Please enter an email message first.")

    else:
        prediction = model.predict([email_text])[0]

        probabilities = model.predict_proba([email_text])[0]
        classes = list(model.classes_)
        spam_index = classes.index("spam")

        spam_probability = probabilities[spam_index] * 100

        st.subheader("Analysis Result")

        if prediction == "spam":
            st.error("🚨 Classification: SPAM")
        else:
            st.success("✅ Classification: NOT SPAM")

        st.metric(
            "Estimated Spam Probability",
            f"{spam_probability:.1f}%"
        )

        st.caption(
            "This probability reflects the experimental model's "
            "estimate, not the actual probability that an email "
            "is malicious. Phishing can also occur in messages "
            "classified as not spam."
        )


st.divider()

st.subheader("How it works")

st.write(
    "1. You enter an email message.\n\n"
    "2. TF-IDF converts the text into numerical features.\n\n"
    "3. A Naive Bayes model predicts the message category."
)

st.caption("Built with Python, Scikit-learn, and Streamlit.")