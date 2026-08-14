# Airline Sentiment Analysis

## Problem Statement

Airlines receive a large volume of customer feedback through social media and online reviews. Manually analyzing these opinions is time-consuming and makes it difficult to quickly identify customer dissatisfaction, service issues, and positive experiences.

**Sentiment Analysis** can automatically classify customer feedback into **Negative, Neutral, or Positive** sentiment, helping organizations understand customer opinions at scale and identify areas that require attention.

---

## Objective

The objective of this project is to build an end-to-end **NLP-based sentiment classification system** for airline customer feedback.

The project focuses on:

- Understanding the linguistic and sentiment patterns present in customer reviews.
- Building a robust text preprocessing pipeline while preserving important sentiment information such as negation.
- Comparing traditional Machine Learning models with a Deep Learning model.
- Evaluating models using **Accuracy, Precision, Recall, Macro-F1, ROC-AUC, and Confusion Matrix**.
- Selecting the strongest model based on overall classification performance.

---

## Dataset

The dataset contains **14,452 customer reviews** with **4 columns**.

### Target Classes

| Class | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

### Class Distribution

| Sentiment | Count | Percentage |
|---|---:|---:|
| Negative | 9,087 | 62.89% |
| Neutral | 3,067 | 21.22% |
| Positive | 2,298 | 15.90% |

The dataset is therefore **imbalanced**, with Negative sentiment representing the majority of observations.
---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

The dataset was explored to understand its structure, quality, and sentiment distribution.

Key EDA steps included:

- Dataset shape and data-type inspection.
- Missing-value analysis.
- Duplicate-text analysis.
- Sentiment/class distribution analysis.
- Distribution of review lengths across the dataset.
- Analysis of class imbalance.
- Investigation of vocabulary size and rare words.
---

### 2. Feature Engineering & Text Preprocessing

A dedicated preprocessing pipeline was developed for the text data.

#### Text Cleaning

- Converted text into a consistent format.
- Removed unnecessary noise from the text.
- Handled URLs, special characters, and irrelevant textual patterns.
- Expanded contractions to preserve their semantic meaning.
- Preserved important negation words such as: `not` `no` `never` `cannot` `without`

#### Train-Test Separation

The dataset was split into training and testing sets before learning vocabulary-based representations to prevent data leakage.

#### Machine Learning Representation

For traditional ML models, text was converted into numerical features using **TF-IDF**.

The final TF-IDF representation used:

- **Unigrams + Bigrams (`ngram_range=(1,2)`)**

#### Deep Learning Representation

For the Deep Learning pipeline:

- Keras Tokenizer was fitted on the training data.
- Text was converted into integer sequences.
- Sequences were padded to a fixed length.
- A learnable embedding representation was used.
- BiLSTM was used to capture sequential and contextual information.

---

## Models Trained

Three traditional Machine Learning models were trained using the same TF-IDF representation:
1. Multinomial Naive Bayes
2. Linear SVM
3. Random Forest
4. BiLSTM