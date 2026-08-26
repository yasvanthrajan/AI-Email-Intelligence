# 📧 AI Email Intelligence

### Machine Learning Powered Email Genre and Emotional Tone Analysis

AI Email Intelligence is a Machine Learning powered application that analyzes email content and automatically identifies:

- 📂 **Email Genre**
- 💭 **Emotional Tone**

The system combines Natural Language Processing, TF-IDF feature extraction, Linear SVM classification, a Transformer-based experiment, and a Flask REST API with a web-based frontend.

---

## 🎯 Project Objective

Emails contain useful information about their purpose and the tone in which they are written.

The objective of this project is to build an intelligent email analysis system that can automatically classify an email into its relevant **genre** and identify its underlying **emotional tone**.

The system takes email text as input and produces human-readable predictions.

---

# 🔄 Project Workflow

```text
                    ┌─────────────────────┐
                    │     Email Input     │
                    │ Subject + Body Text │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Email Cleaning    │
                    │ Subject + Body      │
                    │ Lowercase + Cleanup │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ Genre Analysis   │      │ Emotion Analysis │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ TF-IDF           │      │ TF-IDF           │
        │ Vectorization    │      │ Vectorization    │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ Linear SVM       │      │ Linear SVM       │
        │ Classifier       │      │ Classifier       │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ Genre Prediction │      │ Emotion          │
        │                  │      │ Prediction       │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Flask REST API      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Web Frontend        │
                    │ Human-readable      │
                    │ Results             │
                    └─────────────────────┘
```

---

## 📊 Dataset

The project uses the Enron email dataset with category annotations.

The data loader processes email files and their corresponding `.cats` annotation files.

The project extracts two classification tasks from the annotated categories:

### Email Genre

The genre model uses six top-level `1.x` categories:

| Code | Genre |
| :--- | :--- |
| 1.1 | Company Business, Strategy, etc. |
| 1.2 | Purely Personal |
| 1.3 | Personal but in professional context |
| 1.4 | Logistic Arrangements |
| 1.5 | Employment arrangements |
| 1.6 | Document editing/checking |

### Emotional Tone

The emotional tone model uses nineteen `4.x` categories:

| Code | Emotional Tone |
| :--- | :--- |
| 4.1 | Jubilation |
| 4.2 | Hope / Anticipation |
| 4.3 | Humor |
| 4.4 | Camaraderie |
| 4.5 | Admiration |
| 4.6 | Gratitude |
| 4.7 | Friendship / Affection |
| 4.8 | Sympathy / Support |
| 4.9 | Sarcasm |
| 4.10 | Secrecy / Confidentiality |
| 4.11 | Worry / Anxiety |
| 4.12 | Concern |
| 4.13 | Competitiveness / Aggressiveness |
| 4.14 | Triumph / Gloating |
| 4.15 | Pride |
| 4.16 | Anger / Agitation |
| 4.17 | Sadness / Despair |
| 4.18 | Shame |
| 4.19 | Dislike / Scorn |

---

## 🧹 Email Preprocessing

Before classification, the email goes through a preprocessing stage.

The system:
- Separates email headers and body.
- Extracts the Subject.
- Combines the subject and body.
- Converts the text to lowercase.
- Normalizes whitespace.

This produces the cleaned text used by the machine learning models.

---

## 🧠 Machine Learning Models

### 1. Genre Classification

The genre classification pipeline uses:
`Email Text` → `Text Cleaning` → `TF-IDF Vectorization` → `Linear SVM` → `Multi-label Genre Prediction`

The model supports six genre categories.

#### Genre Model Performance
The Linear SVM model achieved:
- **Micro F1:** 0.6849
- **Macro F1:** 0.5738
- **Micro Precision:** 0.7042
- **Micro Recall:** 0.6667

The genre model performs substantially better than the initial Logistic Regression experiment, particularly across the less frequent categories.

---

### 2. Emotional Tone Classification

The emotional tone pipeline uses:
`Email Text` → `Text Cleaning` → `TF-IDF Vectorization` → `Linear SVM` → `Multi-label Emotional Tone Prediction`

The model supports nineteen emotional-tone categories.

#### Emotional Tone Model Performance
The current Linear SVM model achieved:
- **Micro F1:** 0.4259
- **Macro F1:** 0.1573
- **Micro Precision:** 0.7931
- **Micro Recall:** 0.2911

The emotional dataset is considerably smaller and more imbalanced than the genre dataset, which affects recall across several low-frequency emotional categories.

---

## 🤖 Transformer Experiment

A Transformer-based approach was also explored using **DistilBERT** (`distilbert-base-uncased`).

The Transformer pipeline included:
`Email Text` → `DistilBERT Tokenizer` → `DistilBERT` → `Multi-label Classification`

The experiment used a subset of the dataset for practical local training.

#### Transformer Results
- **Micro F1:** 0.4889
- **Macro F1:** 0.1243
- **Micro Precision:** 0.6667
- **Micro Recall:** 0.3860

The Transformer experiment is included as an additional modeling approach. The final application prediction pipeline uses the trained TF-IDF + Linear SVM models.

---

## 🌐 Application Architecture

```text
┌──────────────────────────────┐
│          Frontend            │
│                              │
│ HTML + CSS + JavaScript      │
└──────────────┬───────────────┘
               │
               │ HTTP POST
               ▼
┌──────────────────────────────┐
│          Flask API           │
│                              │
│ /predict                     │
│ /health                      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       ML Prediction Layer    │
│                              │
│ Genre SVM + TF-IDF           │
│ Emotion SVM + TF-IDF         │
└──────────────────────────────┘
```

---

## 🔌 Flask API

The backend is implemented using Flask.

### Home
`GET /`

**Returns:**
```json
{
  "message": "AI Email Intelligence API is running!",
  "status": "success"
}
```

### Health Check
`GET /health`

**Returns:**
```json
{
  "status": "healthy"
}
```

### Email Prediction
`POST /predict`

**Request:**
```json
{
  "email": "Subject: Confidential business discussion. We need to review the current situation carefully."
}
```

**Response:**
```json
{
  "status": "success",
  "genre": [
    {
      "code": "1.1",
      "name": "Company Business, Strategy, etc."
    }
  ],
  "emotional_tone": [
    {
      "code": "4.10",
      "name": "Secrecy / Confidentiality"
    }
  ]
}
```

---

## 💻 Frontend

The frontend provides a simple interface where users can enter:
- Email subject
- Email body

After clicking **Analyze Email**, the frontend sends the email content to the Flask `/predict` endpoint.

The results are then displayed as:
- 📂 **Email Genre** (e.g., `1.1` - `Company Business, Strategy, etc.`)
- 💭 **Emotional Tone** (e.g., `4.10` - `Secrecy / Confidentiality`)

---

## 📁 Project Structure

```text
AI-Email-Intelligence/
│
├── backend/
│   └── app.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── models/
│   ├── emotion/
│   │   ├── emotion_mlb.pkl
│   │   ├── emotion_svm.pkl
│   │   └── tfidf_vectorizer.pkl
│   │
│   ├── genre_label_binarizer.pkl
│   ├── genre_svm_model.pkl
│   └── genre_tfidf_vectorizer.pkl
│
├── src/
│   ├── analyze_emotions.py
│   ├── data_loader.py
│   ├── predict.py
│   ├── train_emotion_model.py
│   ├── train_genre_model.py
│   └── train_transformer.py
│
├── .gitignore
└── README.md
```

---

## ⚙️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn, Linear SVM, Multi-label classification
- **Natural Language Processing:** TF-IDF, Text preprocessing, MultiLabelBinarizer
- **Deep Learning / Transformer Experiment:** PyTorch, Hugging Face Transformers, DistilBERT
- **Backend:** Flask, Flask-CORS
- **Frontend:** HTML, CSS, JavaScript
- **Data:** Enron Email Dataset with category annotations

---

## 🚀 Installation

Clone the repository:
```bash
git clone https://github.com/yasvanthrajan/AI-Email-Intelligence.git
```

Move into the project:
```bash
cd AI-Email-Intelligence
```

Create a virtual environment:
```bash
python -m venv venv
```

Activate it on Windows:
```bash
venv\Scripts\activate
```

Install the required packages:
```bash
pip install flask flask-cors pandas scikit-learn joblib numpy
```

For the Transformer experiment:
```bash
pip install torch transformers datasets
```

---

## ▶️ Running the Application

Start the Flask backend:
```bash
cd backend
python app.py
```

The API will run at: `http://127.0.0.1:5000`

The frontend can then be opened through the project's frontend setup.

---

## 🧪 Testing the API

Test the health endpoint:
```bash
curl http://127.0.0.1:5000/health
```

Test email prediction:
```bash
curl -X POST http://127.0.0.1:5000/predict ^
-H "Content-Type: application/json" ^
-d "{\"email\":\"Subject: Confidential business discussion. Please keep this information confidential.\"}"
```

---

## 📌 Example

### Input
```text
Subject: Confidential business discussion

We need to review the current situation carefully.
There are concerns regarding the proposed decision.
Please keep this information confidential.
```

### Output
- **Email Genre:** `1.1 - Company Business, Strategy, etc.`
- **Emotional Tone:** `4.10 - Secrecy / Confidentiality`

---

## 📈 Model Comparison

The project experimented with multiple approaches for genre classification.

| Model | Micro F1 | Macro F1 |
| :--- | :--- | :--- |
| Logistic Regression | 0.6225 | 0.2892 |
| Linear SVM | 0.6849 | 0.5738 |
| DistilBERT | 0.4889 | 0.1243 |

Based on the recorded experiments, the Linear SVM provided the strongest genre classification performance among the tested approaches and is used in the final application.

---

## 🔮 Future Improvements

- Increasing the size of the emotional-tone training data.
- Handling class imbalance more effectively.
- Improving recall for low-frequency emotional categories.
- Hyperparameter tuning for the SVM models.
- Exploring improved Transformer fine-tuning strategies.
- Adding confidence scores to predictions.
- Adding richer email analytics and visualizations.
- Deploying the Flask application to a cloud environment.

---

## 👨‍💻 Author

**Yasvanth Rajan**  
GitHub: [https://github.com/yasvanthrajan](https://github.com/yasvanthrajan)

---

## ⭐ Project Summary

AI Email Intelligence demonstrates how Natural Language Processing and Machine Learning can be used to transform raw email content into structured intelligence.

The system analyzes an email, classifies its genre, identifies its emotional tone, and exposes the predictions through a Flask REST API and web interface.
