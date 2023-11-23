import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
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

# Load the pre-trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Create an ML pipeline
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', model)
])

# Initialize session state
if 'spam_count' not in st.session_state:
    st.session_state.spam_count = 0
if 'not_spam_count' not in st.session_state:
    st.session_state.not_spam_count = 0

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Use the ML pipeline to preprocess, vectorize, and predict
    result = pipeline.predict([input_sms])[0]

    # Display the result
    if result == 1:
        st.header("Spam")
        st.session_state.spam_count += 1
    else:
        st.header("Not Spam")
        st.session_state.not_spam_count += 1

    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(['Not Spam', 'Spam'], [st.session_state.not_spam_count, st.session_state.spam_count], color=['blue', 'red'])
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Predictions')

    st.pyplot(fig)
# Define the SVM model
svm_model = SVC()

# Define the parameter grid for grid search
param_grid = {'classifier__C': [0.1, 1, 10, 100], 'classifier__kernel': ['linear']}

# Create a pipeline with TfidfVectorizer and SVM
svm_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', svm_model)
])

# Perform grid search
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Train the final SVM model with the best parameters
final_svm_model = SVC(**best_params)
final_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', final_svm_model)
])
final_pipeline.fit(X_train, y_train)

# Save the final SVM model
pickle.dump(final_pipeline, open('svm_model.pkl', 'wb'))
