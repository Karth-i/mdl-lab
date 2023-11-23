import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download("punkt")
nltk.download('stopwords')


data = pd.read_csv('spam.csv')
X = data['v2']
y = data[v1"]

X = ["This is a positive example", "This is a negative example", "Another positive example"]
y = [1, 0, 1]  # Binary labels (1 for positive, 0 for negative)

# Text preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [ps.stem(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words]
    return " ".join(tokens)

X_preprocessed = [preprocess_text(text) for text in X]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.1, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the text data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Linear SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the model and vectorizer
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(svm_model, open('linear_svm_model.pkl', 'wb'))
