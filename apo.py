import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Load the data
data_frame = pd.read_csv("spam1.csv")
x = data_frame['EmailText']
y = data_frame['Label']

# Train-test split
x_train, y_train = x[:4457], y[:4457]
x_test, y_test = x[4457:], y[4457:]

# Create a pipeline with CountVectorizer and Support Vector Machine
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', SVC())
])

# Define parameters for grid search
tuned_parameters = {
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': [1e-3, 1e-4],
    'clf__C': [1, 10, 100, 1000]
}

# Create a grid search model
model = GridSearchCV(text_clf, tuned_parameters)

# Train the model
model.fit(x_train, y_train)

# Streamlit UI
st.title("Spam Detection App")

# Input text box for user to enter email text
user_input = st.text_area("Enter Email Text:", "")

# Make predictions when the user clicks the "Predict" button
if st.button("Predict"):
    # Use the model to make predictions on the user input
    prediction = model.predict([user_input])

    # Display the prediction result
    st.write("Prediction:", prediction[0])

    # Display the probability of spam
    probability = model.decision_function([user_input])
    st.write("Probability of Spam:", probability[0])

# Display the model performance on the test set
st.subheader("Model Performance on Test Set")
features_test = model.named_steps['vect'].transform(x_test)
accuracy = model.score(x_test, y_test)
st.write("Accuracy:", accuracy)
