from pathlib import Path
import joblib

from data_loader import genre_df, X, y, mlb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

# ==========================================
# 1. DATA SPLITTING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("\nTraining target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# ==========================================
# 2. TF-IDF FEATURE EXTRACTION
# ==========================================
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF transformation complete!")
print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)


# ==========================================
# 3. FIRST MODEL: LOGISTIC REGRESSION
# ==========================================
model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000
    )
)

model.fit(X_train_tfidf, y_train)
print("\nLogistic Regression training completed!")

y_pred = model.predict(X_test_tfidf)
print("\nLogistic Regression predictions generated!")

print("\nFirst 5 actual labels:")
print(y_test[:5])

print("\nFirst 5 predicted labels:")
print(y_pred[:5])

# Evaluate Logistic Regression
micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
micro_precision = precision_score(y_test, y_pred, average="micro", zero_division=0)
micro_recall = recall_score(y_test, y_pred, average="micro", zero_division=0)

print("\n===== LOGISTIC REGRESSION RESULTS =====")
print(f"Micro F1:        {micro_f1:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Micro Precision: {micro_precision:.4f}")
print(f"Micro Recall:    {micro_recall:.4f}")

print("\n===== LOGISTIC REGRESSION PER-LABEL RESULTS =====")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=mlb.classes_,
        zero_division=0
    )
)


# ==========================================
# 4. SECOND MODEL: LINEAR SVM (BALANCED)
# ==========================================
svm_model = OneVsRestClassifier(
    LinearSVC(class_weight="balanced")
)

svm_model.fit(X_train_tfidf, y_train)
print("\nLinear SVM training completed!")

y_pred_svm = svm_model.predict(X_test_tfidf)
print("\nSVM predictions generated!")

# Evaluate SVM
svm_micro_f1 = f1_score(y_test, y_pred_svm, average="micro", zero_division=0)
svm_macro_f1 = f1_score(y_test, y_pred_svm, average="macro", zero_division=0)
svm_precision = precision_score(y_test, y_pred_svm, average="micro", zero_division=0)
svm_recall = recall_score(y_test, y_pred_svm, average="micro", zero_division=0)

print("\n===== LINEAR SVM RESULTS =====")
print(f"Micro F1:        {svm_micro_f1:.4f}")
print(f"Macro F1:        {svm_macro_f1:.4f}")
print(f"Micro Precision: {svm_precision:.4f}")
print(f"Micro Recall:    {svm_recall:.4f}")

print("\n===== LINEAR SVM PER-LABEL RESULTS =====")
print(
    classification_report(
        y_test,
        y_pred_svm,
        target_names=mlb.classes_,
        zero_division=0
    )
)


# ==========================================
# 5. MODEL ARTIFACT SAVING
# ==========================================
MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(exist_ok=True)

# Save the TF-IDF vectorizer
joblib.dump(
    vectorizer,
    MODEL_DIR / "genre_tfidf_vectorizer.pkl"
)

# Save the trained Balanced SVM model
joblib.dump(
    svm_model,
    MODEL_DIR / "genre_svm_model.pkl"
)

# Save the label binarizer / encoder
joblib.dump(
    mlb,
    MODEL_DIR / "genre_label_binarizer.pkl"
)

print("\nBest classical model and artifacts saved successfully!")