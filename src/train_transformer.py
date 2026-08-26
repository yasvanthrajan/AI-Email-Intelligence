from data_loader import genre_df, mlb

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

import torch
from torch.utils.data import Dataset

import numpy as np


# ============================================================
# 1. PREPARE DATASET
# ============================================================

# Get email text
X = genre_df["clean_text"].astype(str).tolist()

# Get multi-label targets
y = mlb.transform(genre_df["genre_labels"])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ------------------------------------------------------------
# Reduce dataset for the first Transformer experiment
# ------------------------------------------------------------

X_train = X_train[:800]
y_train = y_train[:800]

X_test = X_test[:200]
y_test = y_test[:200]


print("Transformer dataset prepared!")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("Number of labels:", y.shape[1])

print("\nLabels:")
print(mlb.classes_)


# ============================================================
# 2. LOAD PRETRAINED TOKENIZER
# ============================================================

MODEL_NAME = "distilbert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


print("\nTokenizer loaded successfully!")
print("Model:", MODEL_NAME)


# Test tokenization
sample_texts = X_train[:2]

tokenized = tokenizer(
    sample_texts,
    padding=True,
    truncation=True,
    max_length=128
)


print("\nTokenization successful!")

print("Input IDs shape:", len(tokenized["input_ids"]))
print("Attention mask shape:", len(tokenized["attention_mask"]))

print("\nFirst email token IDs:")
print(tokenized["input_ids"][0][:20])


# ============================================================
# 3. LOAD PRETRAINED DISTILBERT
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=6,
    problem_type="multi_label_classification"
)


print("\nTransformer model loaded successfully!")

print("Number of output labels:", model.config.num_labels)

print("Problem type:", model.config.problem_type)


# ============================================================
# 4. CREATE PYTORCH DATASET
# ============================================================

class EmailDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=128
    ):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):

        return len(self.texts)


    def __getitem__(self, index):

        text = self.texts[index]

        label = self.labels[index]


        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )


        item = {

            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    label,
                    dtype=torch.float
                )
        }


        return item


# Create training dataset
train_dataset = EmailDataset(
    X_train,
    y_train,
    tokenizer
)


# Create testing dataset
test_dataset = EmailDataset(
    X_test,
    y_test,
    tokenizer
)


print("\nPyTorch datasets created!")

print(
    "Training dataset size:",
    len(train_dataset)
)

print(
    "Testing dataset size:",
    len(test_dataset)
)


# Check one sample
sample = train_dataset[0]


print("\nSample dataset item:")

print(
    "Input IDs shape:",
    sample["input_ids"].shape
)

print(
    "Attention mask shape:",
    sample["attention_mask"].shape
)

print(
    "Labels:",
    sample["labels"]
)


# ============================================================
# 5. EVALUATION METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred


    # Convert logits to probabilities
    probabilities = 1 / (
        1 + np.exp(-logits)
    )


    # Convert probabilities to binary predictions
    predictions = (
        probabilities >= 0.5
    ).astype(int)


    return {

        "micro_f1":
            f1_score(
                labels,
                predictions,
                average="micro",
                zero_division=0
            ),

        "macro_f1":
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0
            ),

        "micro_precision":
            precision_score(
                labels,
                predictions,
                average="micro",
                zero_division=0
            ),

        "micro_recall":
            recall_score(
                labels,
                predictions,
                average="micro",
                zero_division=0
            )
    }


# ============================================================
# 6. TRAINING CONFIGURATION
# ============================================================

training_args = TrainingArguments(

    output_dir="../models/distilbert_genre",

    # Only one epoch for the first experiment
    num_train_epochs=1,

    # Small batch size for CPU
    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    # Standard fine-tuning learning rate
    learning_rate=2e-5,

    weight_decay=0.01,

    # Evaluate after each epoch
    eval_strategy="epoch",

    # Save after each epoch
    save_strategy="epoch",

    # Keep the best model
    load_best_model_at_end=True,

    # Select best model using Macro F1
    metric_for_best_model="macro_f1",

    greater_is_better=True,

    logging_steps=50,

    # Don't send anything to external experiment trackers
    report_to="none"
)


# ============================================================
# 7. CREATE TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics
)


# ============================================================
# 8. TRAIN TRANSFORMER
# ============================================================

print("\n======================================")
print("Starting Transformer training...")
print("======================================")

print("\nTraining samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))
print("Epochs:", 1)
print("Maximum tokens:", 128)
print("Batch size:", 4)


trainer.train()


print("\n======================================")
print("Transformer training completed!")
print("======================================")


# ============================================================
# 9. FINAL EVALUATION
# ============================================================

print("\nEvaluating Transformer model...")

results = trainer.evaluate()


print("\n===== TRANSFORMER RESULTS =====")

print(
    "Micro F1:",
    round(results.get("eval_micro_f1", 0), 4)
)

print(
    "Macro F1:",
    round(results.get("eval_macro_f1", 0), 4)
)

print(
    "Micro Precision:",
    round(results.get("eval_micro_precision", 0), 4)
)

print(
    "Micro Recall:",
    round(results.get("eval_micro_recall", 0), 4)
)


# ============================================================
# 10. SAVE FINAL TRANSFORMER MODEL
# ============================================================

MODEL_OUTPUT_DIR = "../models/distilbert_genre_final"


trainer.save_model(
    MODEL_OUTPUT_DIR
)

tokenizer.save_pretrained(
    MODEL_OUTPUT_DIR
)


print("\nTransformer model saved successfully!")

print(
    "Saved to:",
    MODEL_OUTPUT_DIR
)