import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

# Initialize session state
if 'count_data' not in st.session_state:
    st.session_state.count_data = {'spam': 0, 'not_spam': 0}

# Streamlit UI
st.title("Spam Detection App")

# Input text box for user to enter email text
user_input = st.text_area("Enter Text:", "")

# Make predictions when the user clicks the "Predict" button
if st.button("Predict"):
    # Use the model to make predictions on the user input
    prediction = model.predict([user_input])

    # Display the prediction result
    st.write("Prediction:", prediction[0])

    # Update the count_data based on the prediction
    if prediction[0] == 'spam':
        st.session_state.count_data['spam'] += 1
    else:
        st.session_state.count_data['not_spam'] += 1

    # Display the probability of spam
    probability = model.decision_function([user_input])
    st.write("Probability of Spam:", probability[0])

# Display the model performance on the test set
st.subheader("Model Performance on Test Set")
# Access the feature transformation step directly from the best_estimator_
features_test = model.best_estimator_['vect'].transform(x_test)
accuracy = model.score(x_test, y_test)
st.write("Accuracy:", accuracy)

# Display the bar chart showing the distribution of spam and not spam
st.subheader("Distribution of Spam and Not Spam")
fig, ax = plt.subplots()
ax.bar(st.session_state.count_data.keys(), st.session_state.count_data.values())
ax.set_ylabel('Count')
ax.set_title('Distribution of Spam and Not Spam')
st.pyplot(fig)
