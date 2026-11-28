import re

def clean_text(text):
    """
    Clean SMS and Email message text.
    """

    if not isinstance(text, str):
        return ""

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace URLs with a common token
    text = re.sub(r"http\S+|www\.\S+", " URLTOKEN ", text)

    # Replace email addresses
    text = re.sub(
        r"\S+@\S+\.\S+",
        " EMAILTOKEN ",
        text
    )

    # Remove unwanted special characters
    # Preserve letters and numbers
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text