import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
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

st.title("Email/SMS Spam Classifier")

# Use st.file_uploader to upload a dataset
uploaded_file = st.file_uploader("Upload a dataset in CSV format", type=["csv"])
if uploaded_file is not None:
    # Read the uploaded dataset
    df = pd.read_csv(uploaded_file)

    # Print column names
    st.write("Column Names:", df.columns.tolist())

    # Assuming the dataset has a column 'message' for text input
    input_sms = st.text_area("Enter the message")

    if st.button('Predict'):
        # Use the ML pipeline to preprocess, vectorize, and predict
        result = pipeline.predict([transform_text(input_sms)])[0]

        # Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")

        # Check if 'message' column is in the DataFrame
        if 'message' in df.columns:
            # Data for visualization
            predictions = pipeline.predict([transform_text(message) for message in df['v2']])
            values, counts = np.unique(predictions, return_counts=True)

            # Bar chart
            fig, ax = plt.subplots()
            ax.bar(['Not Spam', 'Spam'], counts, color=['blue', 'red'])
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Predictions')

            st.pyplot(fig)
        else:
            st.warning("The 'message' column is not found in the uploaded dataset.")
