from flask import Flask, render_template, request, jsonify
import joblib
from pathlib import Path
import re

app = Flask(__name__)

# ==========================================
# 1. LOAD MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

model = joblib.load(MODEL_DIR / "spam_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")


# ==========================================
# 2. TEXT CLEANING
# Must match training preprocessing
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ==========================================
# 3. HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# 4. SPAM DETECTION API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True) or {}

        message = data.get("message", "").strip()

        if not message:
            return jsonify({
                "error": "Please enter a message."
            }), 400

        # Clean input before prediction
        cleaned_message = clean_text(message)

        if not cleaned_message:
            return jsonify({
                "error": "Please enter a valid message."
            }), 400

        # Convert cleaned text into TF-IDF
        message_tfidf = vectorizer.transform(
            [cleaned_message]
        )

        # Predict classification
        prediction = model.predict(message_tfidf)[0]

        prediction = str(prediction).lower()

        # ======================================
        # 5. SUSPICIOUS WORD DETECTION
        # ======================================

        suspicious_words = []

        if (
            prediction == "spam"
            and hasattr(model, "feature_log_prob_")
        ):

            classes = [
                str(c).lower()
                for c in model.classes_
            ]

            spam_index = classes.index("spam")
            ham_index = classes.index("ham")

            spam_scores = model.feature_log_prob_[spam_index]
            ham_scores = model.feature_log_prob_[ham_index]

            feature_scores = (
                message_tfidf.toarray()[0]
                * (spam_scores - ham_scores)
            )

            analyzer = vectorizer.build_analyzer()

            words = set(analyzer(cleaned_message))

            vocabulary = vectorizer.vocabulary_

            for word in words:

                if word in vocabulary:

                    word_index = vocabulary[word]

                    contribution = feature_scores[word_index]

                    if contribution > 0:
                        suspicious_words.append(word)

        # ======================================
        # 6. RETURN RESULT
        # ======================================

        return jsonify({

            "prediction": prediction.upper(),

            "trigger_words": suspicious_words,

            "message": message

        })

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# 7. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)
