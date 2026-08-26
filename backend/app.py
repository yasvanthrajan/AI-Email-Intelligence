from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import sys
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"
FRONTEND_DIR = BASE_DIR / "frontend"

sys.path.append(str(SRC_DIR))


# ============================================================
# IMPORT PREDICTION FUNCTION
# ============================================================

from predict import predict_email


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# SERVE FRONTEND
# ============================================================

@app.route("/", methods=["GET"])
def serve_frontend():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================================
# SERVE FRONTEND STATIC FILES
# ============================================================

@app.route("/<path:filename>")
def serve_static(filename):

    file_path = FRONTEND_DIR / filename

    if file_path.exists() and file_path.is_file():

        return send_from_directory(
            FRONTEND_DIR,
            filename
        )

    return jsonify({
        "error": "File not found"
    }), 404


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy"
    })


# ============================================================
# EMAIL PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Get JSON request
        # ----------------------------------------------------

        data = request.get_json()

        if not data:

            return jsonify({
                "status": "error",
                "message": "Request body is required"
            }), 400


        # ----------------------------------------------------
        # Get email text
        # ----------------------------------------------------

        email_text = data.get("email")


        if not email_text:

            return jsonify({
                "status": "error",
                "message": "Email text is required"
            }), 400


        # ----------------------------------------------------
        # Run ML prediction
        # ----------------------------------------------------

        genres, emotions = predict_email(email_text)


        # ----------------------------------------------------
        # Convert predictions to normal Python lists
        # ----------------------------------------------------

        genres = list(genres)
        emotions = list(emotions)


        # ====================================================
        # GENRE LABEL MAPPING
        # ====================================================

        genre_names = {

            "1.1": "Company Business, Strategy, etc.",

            "1.2": "Purely Personal",

            "1.3": "Personal but in professional context",

            "1.4": "Logistic Arrangements",

            "1.5": "Employment arrangements",

            "1.6": "Document editing/checking"
        }


        # ====================================================
        # EMOTIONAL TONE LABEL MAPPING
        # ====================================================

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


        # ====================================================
        # BUILD GENRE RESULTS
        # ====================================================

        genre_results = []

        for genre in genres:

            genre_results.append({

                "code": genre,

                "name": genre_names.get(
                    genre,
                    "Unknown"
                )

            })


        # ====================================================
        # BUILD EMOTION RESULTS
        # ====================================================

        emotion_results = []

        for emotion in emotions:

            emotion_results.append({

                "code": emotion,

                "name": emotion_names.get(
                    emotion,
                    "Unknown"
                )

            })


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return jsonify({

            "status": "success",

            "genre": genre_results,

            "emotional_tone": emotion_results

        })


    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import webbrowser

    url = "http://127.0.0.1:5000/"

    webbrowser.open(url)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )