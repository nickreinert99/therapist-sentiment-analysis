# Therapist Sentiment Analysis

## Overview
This project analyzes therapist responses from an online mental health platform to classify communication tone using sentiment analysis and machine learning models. The goal is to predict whether therapist messages exhibit directive, advisory, or affirmative tones based on textual and linguistic features.

We first explore the relationship between emotional sentiment scores (VADER compound) and directive communication. We then build and evaluate machine learning models to predict sentiment categories (Positive, Neutral, Negative) from therapist responses.

## Methods
- **Text Cleaning and Preprocessing:** Stopword removal, tokenization, lemmatization.
- **Feature Engineering:** 
  - Sentiment labeling (Positive, Negative, Neutral) using VADER and Opinion Lexicon.
  - Tone detection (Directive, Advisory, Affirmative) via keyword matching.
- **Text Vectorization:** 
  - TF-IDF (Term Frequency–Inverse Document Frequency)
  - Bag of Words (BoW)
- **Machine Learning Models:**
  - Decision Tree Classifier (TF-IDF and enhanced features)
  - Logistic Regression (TF-IDF and enhanced features)
  - Naive Bayes Classifier (BoW and enhanced features)

## Project Structure
- `sentiment_analysis_directive_responses_project.ipynb` — Main notebook containing all code, models, evaluation, and visualizations.
- `README.md` — Project overview and instructions.

## Results
- Combining textual features with engineered tone indicators improved model performance across all classifiers.
- Decision Trees and Logistic Regression models showed strong performance when both TF-IDF features and directive/advisory/affirmative indicators were included.
- ROC curve analysis helped visualize model performance across sentiment categories.

## How to Run
1. Clone the repository.
2. Open the `sentiment_analysis_directive_responses_project.ipynb` notebook.
3. Run all cells (data is assumed to be preloaded — otherwise paths need to be adjusted).

## Future Work
- Incorporating deep learning models (e.g., BERT embeddings) for more context-aware analysis.
- Expanding the keyword sets for better tone detection.
- Testing models on larger, more diverse datasets.

## Contact
For any questions, feel free to reach out via GitHub at [nickreinert99](https://github.com/nickreinert99).
