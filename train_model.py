import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "dataset" / "combined_preprocessed.csv"
df = pd.read_csv(dataset_path)

required_columns = {"label", "text"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}"
    )

# Check dataset column names

print("Dataset Columns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())
X = df["text"]
y = df["label"]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Prediction
y_pred = model.predict(X_test_tfidf)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(
    y_test, y_pred, average="weighted", zero_division=0
))
print("Recall:", recall_score(
    y_test, y_pred, average="weighted", zero_division=0
))
print("F1 Score:", f1_score(
    y_test, y_pred, average="weighted", zero_division=0
))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Save trained files
model_directory = project_root / "saved_models"
model_directory.mkdir(exist_ok=True)

joblib.dump(model, model_directory / "spam_model.pkl")
joblib.dump(vectorizer, model_directory / "tfidf_vectorizer.pkl")

print("\nModel training completed successfully!")
print("Model and vectorizer saved!")
# =================================
# STEP 8: CUSTOM MESSAGE TESTING
# =================================

test_messages = [
    "Congratulations! You won a free prize. Claim now!",
    "Hey, are you coming to college today?",
    "You have won cash reward. Call now!",
    "Please submit your assignment tomorrow."
]

# Convert messages into TF-IDF features
test_tfidf = vectorizer.transform(test_messages)

# Predict using trained Naive Bayes model
predictions = model.predict(test_tfidf)

# Display results
print("\n===== CUSTOM MESSAGE TESTING =====")

for message, prediction in zip(test_messages, predictions):

    print("\nMessage:", message)
    print("Prediction:", prediction.upper())