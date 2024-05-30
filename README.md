# Email Spam Detection Project

## Project Overview
In this project, we build an email spam detector using Python and machine learning. The goal of this project is to classify emails as either 'spam' or 'ham' (not spam).

## Dependencies
- Python
- pandas
- scikit-learn

## Dataset
The dataset used in this project is named 'spam.csv'. It contains two main columns:
- 'v1': This column contains the labels (either 'spam' or 'ham').
- 'v2': This column contains the text of the emails.

## Methodology
We use the Naive Bayes algorithm for the machine learning model and TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction.

## Steps
1. **Import necessary libraries**: We import pandas for data manipulation, train_test_split for splitting the data, TfidfVectorizer for feature extraction, and MultinomialNB (Naive Bayes) for the machine learning model.
2. **Load the dataset**: We load the 'spam.csv' dataset using pandas.
3. **Split the dataset into training and testing sets**: We split the 'v2' column (email text) as our feature and the 'v1' column (labels) as our target.
4. **Convert the text data into numerical data using TF-IDF**: We initialize a TfidfVectorizer with English stop words, fit and transform the training data, and transform the testing data.
5. **Create the Naive Bayes model and train it**: We create a MultinomialNB model and train it using the transformed training data.
6. **Make predictions on the testing set and evaluate the model**: We make predictions on the transformed testing set and calculate the accuracy, precision, recall, and F1 score of the model.

## Results
The performance of the model is evaluated using four metrics: Accuracy, Precision, Recall, and F1 Score. The results are printed at the end of the notebook.
