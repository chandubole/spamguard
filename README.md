# spamguard
SMS AND EMAIL SPAM DETECTION
Spam Guard – Data Preprocessing

1. Project Overview

This repository contains the Data Preprocessing component of the Spam Guard project.

The main objective is to clean and prepare SMS and Email datasets for the Machine Learning team to perform feature extraction, model training, and evaluation.

2. Datasets

The project uses two datasets:

- SMS Spam Collection Dataset
- Enron Email Spam Dataset

3. Preprocessing Operations

The following preprocessing operations were performed:

1. Dataset loading
2. Missing value handling
3. Duplicate removal
4. Conversion of text into lowercase
5. HTML tag removal
6. URL and email address handling
7. Special character removal
8. Extra whitespace removal
9. Integration of SMS and Email datasets

4. Repository Files

File Name| Description
"applypreprocessing.py"| Main preprocessing script
"cleandata.py"| Contains text cleaning functions
"validate_data.py"| Data validation script
"cleaned_sms.csv"| Preprocessed SMS dataset
"cleaned_email.zip"| Compressed preprocessed Email dataset
"combined_preprocessed.zip"| Compressed combined dataset

5. Dataset Format

The preprocessed datasets contain the following columns:

- "text" – Cleaned message content
- "label" – Classification label (Spam or Ham)

6. Output Statistics

Dataset| Number of Records
SMS| 5,124
Email| 29,280
Combined| 34,404

Label Distribution

Label| Number of Records
Ham| 19,922
Spam| 14,482

7. How to Use

Install the required Python libraries:

pip install pandas

To execute the preprocessing script:

python applypreprocessing.py

Note: The original datasets must be available in the expected input folders to rerun preprocessing.

8. Next Stage – Machine Learning

The preprocessed datasets are provided to the Machine Learning team for:

1. Feature extraction
2. Text vectorization
3. Model training
4. Model evaluation
5. Spam and Ham classification

9. Team Collaboration

This repository is maintained as part of the Spam Guard project.

The Data Preprocessing component prepares cleaned datasets for the Machine Learning team to continue model development.
