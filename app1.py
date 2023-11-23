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

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Use the ML pipeline to preprocess, vectorize, and predict
    result = pipeline.predict([input_sms])[0]

    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

    # Data for visualization
    #messages = st.text_input("Enter multiple messages separated by commas")
    #predictions = pipeline.predict([" ".join(ps.stem(word.lower()) for word in nltk.word_tokenize(message)) for message in messages.split(",")])

    # Count occurrences using Pandas
    counts_df = pd.Series(predictions).value_counts()

    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(counts_df.index, counts_df.values, color=['blue', 'red'])
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Predictions')

    st.pyplot(fig)
