from data_loader import emotion_df, emotion_mlb

from collections import Counter


print("\n======================================")
print("EMOTIONAL TONE ANALYSIS")
print("======================================")


# Meaning of each emotional category
emotion_names = {
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


# Count labels
counter = Counter()

for labels in emotion_df["emotion_labels"]:

    for label in labels:

        counter[label] += 1


print("\nEmotion distribution:\n")


for label in sorted(
    emotion_names.keys(),
    key=lambda x: int(x.split(".")[1])
):

    print(
        f"{label:5} "
        f"{emotion_names[label]:35} "
        f"{counter[label]:3} emails"
    )


# ============================================================
# MULTI-LABEL CHECK
# ============================================================

multi_label_count = 0

for labels in emotion_df["emotion_labels"]:

    if len(labels) > 1:
        multi_label_count += 1


print("\n======================================")
print("MULTI-LABEL ANALYSIS")
print("======================================")

print(
    "Total emotion emails:",
    len(emotion_df)
)

print(
    "Emails with multiple emotions:",
    multi_label_count
)

print(
    "Emails with single emotion:",
    len(emotion_df) - multi_label_count
)


# ============================================================
# EXAMPLES
# ============================================================

print("\n======================================")
print("EXAMPLES")
print("======================================")


shown = 0

for _, row in emotion_df.iterrows():

    if len(row["emotion_labels"]) > 1:

        print("\nEmail ID:", row["email_id"])

        print(
            "Emotions:",
            [
                emotion_names[label]
                for label in row["emotion_labels"]
            ]
        )

        print(
            "Text:",
            row["clean_text"][:500]
        )

        shown += 1

        if shown == 10:
            break