import joblib
import numpy as np

# Load trained model and vectorizer
model = joblib.load("saved_models/spam_model.pkl")
vectorizer = joblib.load("saved_models/tfidf_vectorizer.pkl")

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Find spam and ham class indexes
spam_index = list(model.classes_).index("spam")
ham_index = list(model.classes_).index("ham")

# Calculate spam-specific word scores
spam_scores = (
    model.feature_log_prob_[spam_index]
    - model.feature_log_prob_[ham_index]
)

# Get top 20 spam-related words
top_indices = np.argsort(spam_scores)[::-1][:20]

print("\n===== TOP SPAM TRIGGER WORDS =====")

for index in top_indices:
    print(
        feature_names[index],
        ":",
        round(spam_scores[index], 4)
    )