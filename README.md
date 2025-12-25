# Zomato Review Sentiment Analysis

## Project Overview
This project aims to perform sentiment analysis on Zomato restaurant reviews using a Gaussian Naive Bayes classifier. The goal is to classify reviews as 'Liked' (positive sentiment) or 'Not Liked' (negative sentiment) based on their textual content.

## Dataset
The dataset used is `Dataset_master.csv`, which contains two columns:
- `Review`: The textual content of the Zomato review.
- `Liked`: A binary label (1 for liked, 0 for not liked) indicating the sentiment of the review.

## Methodology
The project follows a standard machine learning pipeline for natural language processing:

1.  **Library Imports**: Essential libraries such as `numpy`, `pandas`, `matplotlib.pyplot`, `re`, `nltk`, `sklearn` were imported.
2.  **Data Loading**: The `Dataset_master.csv` file was loaded into a pandas DataFrame.
3.  **Text Preprocessing**: Each review underwent several cleaning steps:
    *   Removal of non-alphabetic characters.
    *   Conversion to lowercase.
    *   Tokenization (splitting into words).
    *   Removal of common English stopwords (with 'not' retained as it often reverses sentiment).
    *   Stemming using the Porter Stemmer to reduce words to their root form.
    *   Handling of potential `NaN` values in the 'Review' column by converting them to strings.
4.  **Feature Extraction (Bag of Words)**: The preprocessed text reviews were converted into a numerical format using `CountVectorizer`. This created a Bag of Words model, representing each review as a vector of word frequencies. `max_features` was set to 1600.
5.  **Target Variable Preparation**: The 'Liked' column was extracted as the target variable `y`.
6.  **Handling Missing Values**: Rows containing `NaN` values in the target variable `y` were removed from both the feature matrix `x` and the target vector `y` to ensure data consistency.
7.  **Data Splitting**: The dataset was divided into training and testing sets using `train_test_split`, with 80% for training and 20% for testing (`random_state=42`).
8.  **Model Training**: A Gaussian Naive Bayes classifier (`GaussianNB`) was trained on the preprocessed training data.
9.  **Prediction**: The trained classifier was used to predict sentiment labels (`y_pred`) on the unseen test data.
10. **Model Evaluation**: The model's performance was evaluated using a confusion matrix and accuracy score.

## Results

### Confusion Matrix
```
[[48 48]
 [18 86]]
```
- **True Negatives (TN)**: 48 reviews were correctly predicted as 'Not Liked'.
- **False Positives (FP)**: 48 reviews were incorrectly predicted as 'Liked' (but were actually 'Not Liked').
- **False Negatives (FN)**: 18 reviews were incorrectly predicted as 'Not Liked' (but were actually 'Liked').
- **True Positives (TP)**: 86 reviews were correctly predicted as 'Liked'.

### Accuracy Score
```
0.67
```
The model achieved an accuracy of 67% on the test set.
