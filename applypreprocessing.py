import pandas as pd
import re
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

SMS_PATH = BASE_DIR / "dataset" / "sms" / "SMSSpamCollection"
EMAIL_PATH = BASE_DIR / "dataset" / "e mail" / "enron_spam_data.csv"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):

    text = str(text)

    # Convert text into lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove special characters
    # Keep alphabets and numbers
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ==========================================
# 3. LABEL CLEANING FUNCTION
# ==========================================

def clean_label(label):

    label = str(label).strip().lower()

    if label == "spam":
        return "spam"

    elif label == "ham":
        return "ham"

    else:
        raise ValueError(f"Unknown label found: {label}")


# ==========================================
# 4. SMS DATASET PREPROCESSING
# ==========================================

def preprocess_sms():

    print("\nStarting SMS preprocessing...")

    if not SMS_PATH.exists():
        raise FileNotFoundError(
            f"SMS file not found: {SMS_PATH}"
        )

    # Read SMS dataset
    sms = pd.read_csv(
        SMS_PATH,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8"
    )

    print("SMS dataset loaded successfully!")

    # Remove missing values
    sms = sms.dropna(subset=["label", "text"])

    # Clean labels
    sms["label"] = sms["label"].apply(clean_label)

    # Clean SMS messages
    sms["text"] = sms["text"].apply(clean_text)

    # Remove empty messages
    sms = sms[sms["text"] != ""]

    # Remove duplicate records
    sms = sms.drop_duplicates()

    # Reset index
    sms = sms.reset_index(drop=True)

    # Save cleaned SMS dataset
    sms.to_csv(
        OUTPUT_DIR / "cleaned_sms.csv",
        index=False
    )

    print("SMS preprocessing completed!")
    print("SMS records:", len(sms))

    return sms


# ==========================================
# 5. EMAIL DATASET PREPROCESSING
# ==========================================

def preprocess_email():

    print("\nStarting Email preprocessing...")

    if not EMAIL_PATH.exists():
        raise FileNotFoundError(
            f"Email file not found: {EMAIL_PATH}"
        )

    # Read email CSV
    email = pd.read_csv(
        EMAIL_PATH,
        encoding="latin-1"
    )

    print("Email dataset loaded successfully!")

    # Display original columns
    print("Email columns:", email.columns.tolist())

    # Identify label column
    label_candidates = [
        "label", "spam/ham", "spam_ham",
        "class", "category", "target"
    ]

    label_col = None

    for col in email.columns:
        if col.strip().lower() in label_candidates:
            label_col = col
            break

    # Identify message column
    text_candidates = [
        "message", "text", "body",
        "content", "email"
    ]

    text_col = None

    for col in email.columns:
        if col.strip().lower() in text_candidates:
            text_col = col
            break

    # Check columns
    if label_col is None or text_col is None:
        raise ValueError(
            "Could not identify email columns. "
            f"Available columns: {email.columns.tolist()}"
        )

    # Select required columns
    email = email[[label_col, text_col]].copy()

    # Rename columns
    email.columns = ["label", "text"]

    # Remove missing values
    email = email.dropna(subset=["label", "text"])

    # Clean labels
    email["label"] = email["label"].apply(clean_label)

    # Clean email messages
    email["text"] = email["text"].apply(clean_text)

    # Remove empty messages
    email = email[email["text"] != ""]

    # Remove duplicates
    email = email.drop_duplicates()

    # Reset index
    email = email.reset_index(drop=True)

    # Save cleaned email dataset
    email.to_csv(
        OUTPUT_DIR / "cleaned_email.csv",
        index=False
    )

    print("Email preprocessing completed!")
    print("Email records:", len(email))

    return email


# ==========================================
# 6. COMBINE BOTH DATASETS
# ==========================================

def combine_datasets(sms, email):

    print("\nCombining SMS and Email datasets...")

    combined = pd.concat(
        [sms, email],
        ignore_index=True
    )

    # Remove duplicate records
    combined = combined.drop_duplicates()

    # Reset index
    combined = combined.reset_index(drop=True)

    # Save combined dataset
    combined.to_csv(
        OUTPUT_DIR / "combined_preprocessed.csv",
        index=False
    )

    print("Both datasets combined successfully!")

    return combined


# ==========================================
# 7. FINAL SUMMARY
# ==========================================

def main():

    print("\n================================")
    print(" SPAM GUARD DATA PREPROCESSING")
    print("================================")

    sms = preprocess_sms()

    email = preprocess_email()

    combined = combine_datasets(sms, email)

    print("\n================================")
    print(" DATA PREPROCESSING COMPLETED!")
    print("================================")

    print("\nSMS records:", len(sms))
    print("Email records:", len(email))
    print("Total records:", len(combined))

    print("\nLabel distribution:")
    print(combined["label"].value_counts())

    print("\nOutput files created:")
    print("1. output/cleaned_sms.csv")
    print("2. output/cleaned_email.csv")
    print("3. output/combined_preprocessed.csv")


# ==========================================
# 8. RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    main()