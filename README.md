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
