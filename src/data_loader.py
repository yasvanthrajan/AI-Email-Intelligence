from pathlib import Path
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer


# ============================================================
# 1. DATASET PATH
# ============================================================

DATASET_PATH = Path("../dataset/enron_with_categories")

records = []


# ============================================================
# 2. LOAD EMAILS + .CATS FILES
# ============================================================

# Go through folders 1 to 8
for folder in range(1, 9):

    folder_path = DATASET_PATH / str(folder)

    # Look at every file inside the folder
    for file_path in folder_path.iterdir():

        # Skip annotation files
        if file_path.suffix == ".cats":
            continue

        # Read email
        email_text = file_path.read_text(
            encoding="latin-1",
            errors="ignore"
        )

        # Corresponding .cats file
        cats_path = file_path.with_suffix(".cats")

        # Read category annotations
        cats_text = cats_path.read_text(
            encoding="latin-1",
            errors="ignore"
        )

        records.append({
            "email_id": f"{folder}/{file_path.name}",
            "text": email_text,
            "labels": cats_text.strip()
        })


# Convert to DataFrame
df = pd.DataFrame(records)


print("Dataset loaded successfully!")
print("Number of emails:", len(df))


print("\nFirst 5 records:")
print(df.head())


# ============================================================
# 3. EXTRACT GENRE LABELS (1.x)
# ============================================================

def extract_genre_labels(labels):

    genre_labels = []

    for line in labels.splitlines():

        parts = line.split(",")

        if len(parts) != 3:
            continue

        top_level = parts[0]
        subcategory = parts[1]

        # Keep only 1.x categories
        if top_level == "1":
            genre_labels.append(f"1.{subcategory}")

    return genre_labels


df["genre_labels"] = df["labels"].apply(extract_genre_labels)


# ============================================================
# 4. CLEAN EMAIL
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

    subject = (
        subject_match.group(1)
        if subject_match
        else ""
    )

    # Combine subject + body
    text = subject + " " + body

    # Lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_text"] = df["text"].apply(clean_email)


# ============================================================
# 5. GENRE DATASET
# ============================================================

genre_df = df[
    ["email_id", "clean_text", "genre_labels"]
].copy()


print("\nGenre dataset:")
print(genre_df.head())


print(
    "\nNumber of emails with genre labels:",
    (genre_df["genre_labels"].apply(len) > 0).sum()
)


# ============================================================
# 6. ORIGINAL + CLEANED EMAIL PREVIEW
# ============================================================

print("\nOriginal email:")
print(df["text"].iloc[0][:1000])


print("\nCleaned email:")
print(df["clean_text"].iloc[0][:1000])


# ============================================================
# 7. GENRE LABEL FILTERING
# ============================================================

allowed_genres = {
    "1.1",
    "1.2",
    "1.3",
    "1.4",
    "1.5",
    "1.6"
}


# Keep only selected genre labels
genre_df["genre_labels"] = genre_df[
    "genre_labels"
].apply(
    lambda labels: [
        label
        for label in labels
        if label in allowed_genres
    ]
)


# Remove emails with no genre labels
genre_df = genre_df[
    genre_df["genre_labels"].apply(len) > 0
].reset_index(drop=True)


# ============================================================
# 8. GENRE MULTI-LABEL ENCODING
# ============================================================

mlb = MultiLabelBinarizer(
    classes=sorted(allowed_genres)
)

y = mlb.fit_transform(
    genre_df["genre_labels"]
)


X = genre_df["clean_text"]


print("\n======================================")
print("GENRE ML DATASET PREPARED")
print("======================================")

print("Number of emails:", len(X))
print("Number of labels:", len(mlb.classes_))

print("\nGenre Labels:")
print(mlb.classes_)

print("\nFirst 5 genre target rows:")
print(y[:5])


# ============================================================
# 9. EXTRACT EMOTIONAL TONE LABELS (4.x)
# ============================================================

def extract_emotion_labels(labels):

    emotion_labels = []

    for line in labels.splitlines():

        parts = line.split(",")

        if len(parts) != 3:
            continue

        top_level = parts[0]
        subcategory = parts[1]

        # Keep only 4.x categories
        if top_level == "4":
            emotion_labels.append(
                f"4.{subcategory}"
            )

    return emotion_labels


df["emotion_labels"] = df[
    "labels"
].apply(extract_emotion_labels)


# ============================================================
# 10. CREATE EMOTION DATASET
# ============================================================

emotion_df = df[
    ["email_id", "clean_text", "emotion_labels"]
].copy()


print("\n======================================")
print("EMOTIONAL TONE DATASET")
print("======================================")

print("\nFirst 5 emotion records:")
print(emotion_df.head())


print(
    "\nNumber of emails with emotion labels:",
    (
        emotion_df["emotion_labels"]
        .apply(len) > 0
    ).sum()
)


# ============================================================
# 11. EMOTION LABELS
# ============================================================

allowed_emotions = {
    "4.1",
    "4.2",
    "4.3",
    "4.4",
    "4.5",
    "4.6",
    "4.7",
    "4.8",
    "4.9",
    "4.10",
    "4.11",
    "4.12",
    "4.13",
    "4.14",
    "4.15",
    "4.16",
    "4.17",
    "4.18",
    "4.19"
}


# Keep only valid emotion labels
emotion_df["emotion_labels"] = emotion_df[
    "emotion_labels"
].apply(
    lambda labels: [
        label
        for label in labels
        if label in allowed_emotions
    ]
)


# Remove emails without emotion labels
emotion_df = emotion_df[
    emotion_df["emotion_labels"].apply(len) > 0
].reset_index(drop=True)


# ============================================================
# 12. EMOTION MULTI-LABEL ENCODING
# ============================================================

emotion_mlb = MultiLabelBinarizer(
    classes=sorted(allowed_emotions)
)


emotion_y = emotion_mlb.fit_transform(
    emotion_df["emotion_labels"]
)


emotion_X = emotion_df["clean_text"]


# ============================================================
# 13. EMOTION DATASET SUMMARY
# ============================================================

print("\n======================================")
print("EMOTION ML DATASET PREPARED")
print("======================================")

print("Number of emails:", len(emotion_X))

print(
    "Number of emotion labels:",
    len(emotion_mlb.classes_)
)


print("\nEmotion Labels:")
print(emotion_mlb.classes_)


print("\nFirst 5 emotion target rows:")
print(emotion_y[:5])


# ============================================================
# 14. EMOTION LABEL DISTRIBUTION
# ============================================================

print("\n======================================")
print("EMOTION LABEL DISTRIBUTION")
print("======================================")


for index, label in enumerate(
    emotion_mlb.classes_
):

    count = emotion_y[:, index].sum()

    print(
        f"{label}: {count}"
    )


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n======================================")
print("DATA LOADER COMPLETE")
print("======================================")

print(
    "Genre dataset:",
    len(genre_df),
    "emails"
)

print(
    "Genre labels:",
    len(mlb.classes_)
)

print(
    "Emotion dataset:",
    len(emotion_df),
    "emails"
)

print(
    "Emotion labels:",
    len(emotion_mlb.classes_)
)