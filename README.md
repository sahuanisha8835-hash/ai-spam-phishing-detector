# 🛡️ AI Spam & Phishing Email Detector

An introductory machine learning project that classifies text messages as spam or legitimate (ham) using Natural Language Processing (NLP).

## 🚀 Features

* Classifies messages as SPAM or NOT SPAM.
* Estimates the model's spam probability.
* Interactive web interface built with Streamlit.
* Text processing using TF-IDF.
* Machine learning classification using Multinomial Naive Bayes.
* Displays a warning about the model's limitations.

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Multinomial Naive Bayes
* Streamlit
* Joblib

## 📁 Project Structure

```text
ai-spam-phishing-detector/
├── data/
│   └── emails.csv
├── models/
│   └── spam_model.pkl
├── src/
│   ├── check_data.py
│   ├── train_model.py
│   └── predict.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ai-spam-phishing-detector
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train_model.py
```

### 5. Run the application

```bash
streamlit run app.py
```

## 🧠 How It Works

1. The user enters an email message.
2. TF-IDF converts the text into numerical features.
3. Multinomial Naive Bayes predicts whether the message is spam or legitimate.
4. Streamlit displays the classification and estimated spam probability.

## ⚠️ Limitations

This is an educational prototype trained on a small sample dataset of 20 messages. Its accuracy and predictions are not reliable enough for real-world security decisions. It has not been validated against a representative phishing-email dataset.

## 👩‍💻 Author

**Anisha Sahu**
B.Tech Computer Science Engineering Student

---

Built as a hands-on project to learn machine learning, NLP, and Python web application development.
