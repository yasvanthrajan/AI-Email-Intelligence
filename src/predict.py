import re
import joblib
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# ============================================================
# LOAD GENRE MODEL
# ============================================================

genre_vectorizer = joblib.load(
    MODEL_DIR / "genre_tfidf_vectorizer.pkl"
)

genre_model = joblib.load(
    MODEL_DIR / "genre_svm_model.pkl"
)

genre_mlb = joblib.load(
    MODEL_DIR / "genre_label_binarizer.pkl"
)


# ============================================================
# LOAD EMOTION MODEL
# ============================================================

emotion_vectorizer = joblib.load(
    MODEL_DIR / "emotion" / "tfidf_vectorizer.pkl"
)

emotion_model = joblib.load(
    MODEL_DIR / "emotion" / "emotion_svm.pkl"
)

emotion_mlb = joblib.load(
    MODEL_DIR / "emotion" / "emotion_mlb.pkl"
)


print("======================================")
print("MODELS LOADED SUCCESSFULLY")
print("======================================")

print("Genre labels:")
print(genre_mlb.classes_)

print("\nEmotion labels:")
print(emotion_mlb.classes_)


# ============================================================
# HUMAN-READABLE LABEL MAPPINGS
# ============================================================

GENRE_NAMES = {
    "1.1": "Company Business, Strategy, etc.",
    "1.2": "Purely Personal",
    "1.3": "Personal but in professional context",
    "1.4": "Logistic Arrangements",
    "1.5": "Employment arrangements",
    "1.6": "Document editing/checking"
}

EMOTION_NAMES = {
    "4.1": "Jubilation",
    "4.2": "Hope / Anticipation",
    "4.3": "Humor",
    "4.4": "Camaraderie",
    "4.5": "Admiration",
    "4.6": "Gratitude",
    "4.7": "Friendship / Affection",
    "4.8": "Sympathy / Support",
    "4.9": "Sarcasm",
    "4.10": "Secrecy / Confidentiality",
    "4.11": "Worry / Anxiety",
    "4.12": "Concern",
    "4.13": "Competitiveness / Aggressiveness",
    "4.14": "Triumph / Gloating",
    "4.15": "Pride",
    "4.16": "Anger / Agitation",
    "4.17": "Sadness / Despair",
    "4.18": "Shame",
    "4.19": "Dislike / Scorn"
}


# ============================================================
# EMAIL CLEANING
# ============================================================

def clean_email(email_text):

    # Split headers and body
    parts = email_text.split("\n\n", 1)

    if len(parts) == 2:
        headers = parts[0]
        body = parts[1]
    else:
        headers = email_text
        body = ""

    # Extract subject
    subject_match = re.search(
        r"^Subject:\s*(.*)$",
        headers,
        re.MULTILINE | re.IGNORECASE
    )

    subject = subject_match.group(1) if subject_match else ""

    # Combine subject + body
    text = subject + " " + body

    # Lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_email(email_text):

    # --------------------------------------------------------
    # CLEAN EMAIL
    # --------------------------------------------------------

    clean_text = clean_email(email_text)

    # ========================================================
    # GENRE PREDICTION
    # ========================================================

    genre_features = genre_vectorizer.transform(
        [clean_text]
    )

    genre_prediction = genre_model.predict(
        genre_features
    )

    predicted_genres = genre_mlb.inverse_transform(
        genre_prediction
    )[0]

    # ========================================================
    # EMOTION PREDICTION
    # ========================================================

    emotion_features = emotion_vectorizer.transform(
        [clean_text]
    )

    # Get SVM decision scores
    emotion_scores = emotion_model.decision_function(
        emotion_features
    )[0]

    # Default SVM predictions
    emotion_prediction = emotion_model.predict(
        emotion_features
    )

    predicted_emotions = emotion_mlb.inverse_transform(
        emotion_prediction
    )[0]

    # ========================================================
    # DEBUG: SHOW EMOTION SCORES
    # ========================================================

    print("\n======================================")
    print("EMOTION DECISION SCORES")
    print("======================================")

    for label, score in zip(
        emotion_mlb.classes_,
        emotion_scores
    ):

        print(
            f"{label:>5} : {score:.4f}"
        )

    print("======================================")

    return predicted_genres, predicted_emotions


# ============================================================
# TEST EMAIL
# ============================================================

email = """
Subject: Confidential business discussion

We need to review the current situation carefully.
There are concerns regarding the proposed decision.
Please keep this information confidential until we
complete our discussion.
"""


# ============================================================
# RUN PREDICTION
# ============================================================

genres, emotions = predict_email(email)


# ============================================================
# HUMAN-READABLE RESULT
# ============================================================

print("\n======================================")
print("EMAIL INTELLIGENCE RESULT")
print("======================================")

print("\nPredicted Genre:")

if len(genres) > 0:
    for genre in genres:
        print(f"- {genre}: {GENRE_NAMES.get(genre, 'Unknown')}")
else:
    print("No genre detected by the classifier.")


print("\nPredicted Emotional Tone:")

if len(emotions) > 0:
    for emotion in emotions:
        print(f"- {emotion}: {EMOTION_NAMES.get(emotion, 'Unknown')}")
else:
    print("No emotional tone detected by the classifier.")

print("\n======================================")