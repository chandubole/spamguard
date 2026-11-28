
from flask import Flask, render_template, request, jsonify
import joblib
from pathlib import Path
import re

app = Flask(__name__)

# 1. Load trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

model = joblib.load(MODEL_DIR / "spam_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")


# 2. Home page
@app.route("/")
def home():
    return render_template("index.html")


# 3. Spam detection API
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True) or {}

        message = data.get("message", "").strip()

        if not message:
            return jsonify({
                "error": "Please enter a message."
            }), 400

        # Convert message into TF-IDF features
        message_tfidf = vectorizer.transform([message])

        # Predict spam or ham
        prediction = model.predict(message_tfidf)[0]

        # 4. Suspicious word detection
        words = re.findall(r"\b[a-zA-Z]+\b", message.lower())

        vocabulary = vectorizer.vocabulary_
        suspicious_words = []

        # Find words that contribute positively
        # to the spam classification
        if str(prediction).lower() == "spam":

            spam_index = list(model.classes_).index("spam")

            feature_scores = (
                message_tfidf.toarray()[0] *
                model.feature_log_prob_[spam_index]
            )

            for word in dict.fromkeys(words):

                if word in vocabulary:

                    word_index = vocabulary[word]

                    ham_index = list(model.classes_).index("ham")

                    spam_score = (
                        model.feature_log_prob_[spam_index][word_index]
                    )

                    ham_score = (
                        model.feature_log_prob_[ham_index][word_index]
                    )

                    if spam_score > ham_score:
                        suspicious_words.append(word)

        # 5. Return result to frontend
        return jsonify({
            "prediction": str(prediction).upper(),
            "trigger_words": suspicious_words,
            "message": message
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# 6. Run Flask application
if __name__ == "__main__":
    app.run(debug=True)
