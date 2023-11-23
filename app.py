import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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