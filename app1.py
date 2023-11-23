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

# Load the pre-trained vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Use linear SVM with hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100]}
svm_model = GridSearchCV(SVC(kernel='linear'), param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# Create an ML pipeline
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', svm_model)
])

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Fit the vectorizer on some data (you should use your training data)
    # Here, we use a placeholder DataFrame for demonstration purposes
    placeholder_data = pd.DataFrame({'message': ['example message'], 'label': [0]})
    tfidf.fit(placeholder_data['message'])

    # Transform the input message using the fitted vectorizer
    input_sms_transformed = tfidf.transform([input_sms])

    # Fit the SVM on some data (you should use your training data)
    # Here, we use a placeholder DataFrame for demonstration purposes
    svm_model.fit(input_sms_transformed, placeholder_data['label'])

    # Use the ML pipeline to predict
    result = pipeline.predict([input_sms])[0]

    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(['Not Spam', 'Spam'], [1 - result, result], color=['blue', 'red'])
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Predictions')

    st.pyplot(fig)
