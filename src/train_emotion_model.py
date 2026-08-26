from data_loader import emotion_df, emotion_mlb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

import numpy as np


# ============================================================
# 1. PREPARE DATA
# ============================================================

X = emotion_df["clean_text"].astype(str).tolist()

y = emotion_mlb.transform(
    emotion_df["emotion_labels"]
)


print("\n======================================")
print("EMOTIONAL TONE DATASET")
print("======================================")

print("Number of emails:", len(X))
print("Number of labels:", y.shape[1])

print("\nEmotion labels:")
print(emotion_mlb.classes_)


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n======================================")
print("TRAIN / TEST SPLIT")
print("======================================")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# ============================================================
# 3. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


print("\n======================================")
print("TF-IDF TRANSFORMATION")
print("======================================")

print(
    "Training feature shape:",
    X_train_tfidf.shape
)

print(
    "Testing feature shape:",
    X_test_tfidf.shape
)


# ============================================================
# 4. LINEAR SVM
# ============================================================

print("\n======================================")
print("TRAINING LINEAR SVM")
print("======================================")


model = OneVsRestClassifier(
    LinearSVC(
        class_weight="balanced",
        random_state=42
    )
)


model.fit(
    X_train_tfidf,
    y_train
)


print("\nLinear SVM training completed!")


# ============================================================
# 5. PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)


print("\nPredictions generated!")


print("\nFirst 5 actual labels:")
print(y_test[:5])


print("\nFirst 5 predicted labels:")
print(y_pred[:5])


# ============================================================
# 6. OVERALL EVALUATION
# ============================================================

micro_f1 = f1_score(
    y_test,
    y_pred,
    average="micro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

micro_precision = precision_score(
    y_test,
    y_pred,
    average="micro",
    zero_division=0
)

micro_recall = recall_score(
    y_test,
    y_pred,
    average="micro",
    zero_division=0
)


print("\n======================================")
print("EMOTIONAL TONE SVM RESULTS")
print("======================================")

print(
    f"Micro F1:        {micro_f1:.4f}"
)

print(
    f"Macro F1:        {macro_f1:.4f}"
)

print(
    f"Micro Precision: {micro_precision:.4f}"
)

print(
    f"Micro Recall:    {micro_recall:.4f}"
)


# ============================================================
# 7. PER-LABEL RESULTS
# ============================================================

print("\n======================================")
print("PER-LABEL RESULTS")
print("======================================")


report = classification_report(
    y_test,
    y_pred,
    target_names=emotion_mlb.classes_,
    zero_division=0
)

print(report)


# ============================================================
# 8. PREDICT EMOTION LABEL NAMES
# ============================================================

print("\n======================================")
print("SAMPLE EMOTION PREDICTIONS")
print("======================================")


for i in range(min(10, len(X_test))):

    predicted_labels = [
        emotion_mlb.classes_[j]
        for j in range(len(emotion_mlb.classes_))
        if y_pred[i][j] == 1
    ]

    actual_labels = [
        emotion_mlb.classes_[j]
        for j in range(len(emotion_mlb.classes_))
        if y_test[i][j] == 1
    ]

    print("\nEmail", i + 1)

    print("Actual:", actual_labels)

    print("Predicted:", predicted_labels)


# ============================================================
# 9. SAVE MODEL
# ============================================================

import os
import joblib


os.makedirs(
    "../models/emotion",
    exist_ok=True
)


joblib.dump(
    vectorizer,
    "../models/emotion/tfidf_vectorizer.pkl"
)


joblib.dump(
    model,
    "../models/emotion/emotion_svm.pkl"
)


joblib.dump(
    emotion_mlb,
    "../models/emotion/emotion_mlb.pkl"
)


print("\n======================================")
print("MODEL SAVED")
print("======================================")

print(
    "Saved vectorizer to:",
    "../models/emotion/tfidf_vectorizer.pkl"
)

print(
    "Saved model to:",
    "../models/emotion/emotion_svm.pkl"
)

print(
    "Saved label encoder to:",
    "../models/emotion/emotion_mlb.pkl"
)


print("\n======================================")
print("EMOTIONAL TONE MODEL COMPLETE")
print("======================================")