import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
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

# Use linear SVM with hyperparameter tuning and Leave-One-Out cross-validation
param_grid = {'classifier__C': [0.1, 1, 10, 100]}
svm_model = SVC(kernel='linear')

# Create an ML pipeline
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', svm_model)
])

# Initialize session state
if 'spam_count' not in st.session_state:
    st.session_state.spam_count = 0
if 'not_spam_count' not in st.session_state:
    st.session_state.not_spam_count = 0

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Fit the vectorizer with placeholder messages
    tfidf.fit(['example message 1', 'example message 2'])

    # Transform the input message using the fitted vectorizer
    input_sms_transformed = tfidf.transform([input_sms])

    # Fit the SVM model on the placeholder dataset
    placeholder_data = pd.DataFrame({
        'message': ['example message 1', 'example message 2'],
        'label': [0, 1]
    })

    # Use Leave-One-Out cross-validation for training
    loo = LeaveOneOut()
    predictions = []
    for train_index, test_index in loo.split(placeholder_data):
        X_train, X_test = input_sms_transformed[train_index], input_sms_transformed[test_index]
        y_train, y_test = placeholder_data['label'].iloc[train_index], placeholder_data['label'].iloc[test_index]

        # Fit the SVM model
        svm_model.fit(X_train, y_train)

        # Predict on the test sample
        result = svm_model.predict(X_test)[0]
        predictions.append(result)

    # Use the ML pipeline to predict
    result = predictions[0]

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
