from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Class LogisticRegressionScratch moved here so pickle can load the trained model without errors
class LogisticRegressionScratch:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X = X.toarray()
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        return [1 if i > 0.5 else 0 for i in y_pred]


# 'punkt' is needed for word_tokenize.
# 'stopwords' is needed for the list of English stop words.
nltk.download("punkt")
nltk.download("stopwords")


# Load the pre-trained Random Forest model and the TF-IDF vectorizer.

with open("rf_model.pkl", "rb") as model_file:
    lr_model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as model_file:
    tfidf_vectorizer = pickle.load(model_file)
    

        
# Define preprocessing function (Must match training pipeline)
def preprocess_text(text):
    
    text = text.lower() #Convert to lowercase
    
    text = re.sub(r'<[^>]*>', '', text) #Remove HTML tags
    
    tokens = word_tokenize(text) #Tokenize the text into individual words
    
    tokens = [word for word in tokens if word.isalnum()] #Remove special characters and keep only alphanumeric tokens

    stop_words_list = set(stopwords.words("english"))#Remove stop words (common words like 'a', 'the', 'is')
    tokens = [word for word in tokens if word not in stop_words_list]

    ps = PorterStemmer() #Apply stemming (reducing words to their root form)
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")
    preprocessed_text = preprocess_text(message)
    
    # Transform the cleaned text into numerical features using the loaded tfidf vectorizer
    # The input must be a list of strings: [cleaned]
    vectorized = tfidf_vectorizer.transform([preprocessed_text])
    
    #Make the prediction using the Random Forest model rf_model
    prediction = lr_model.predict(vectorized)[0]
    
    #Convert the numerical prediction (ham 0 or spam 1)
    label = "spam" if prediction == 1 else "ham"
    
    #return prediction result as a json response
    return jsonify({"prediction": label})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
