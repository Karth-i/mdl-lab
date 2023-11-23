import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd

nltk.download("punkt")
nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Create an empty TfidfVectorizer
tfidf = TfidfVectorizer()

# Use linear SVM with hyperparameter tuning
param_grid = {'classifier__C': [0.1, 1, 10, 100]}
svm_model = GridSearchCV(SVC(kernel='linear'), param_grid, cv=2, scoring='accuracy', n_jobs=-1)

# Create an ML pipeline
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', svm_model)
])

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Fit the vectorizer with a placeholder m
