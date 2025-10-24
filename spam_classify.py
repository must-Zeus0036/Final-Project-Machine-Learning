from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK data
# 'punkt' is needed for word_tokenize.
# 'stopwords' is needed for the list of English stop words.
nltk.download("punkt")
nltk.download("stopwords")


# Load the pre-trained Random Forest model and the TF-IDF vectorizer.
with open("rf_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
    

# Define preprocessing function (Must match training pipeline)
def preprocess_text(text):
    """
    Applies the exact same cleaning, tokenization, stop word removal,
    and stemming steps from model
    """
    text = text.lower() #Convert to lowercase
    
    text = re.sub(r'<[^>]*>', '', text) #Remove HTML tags
    
    tokens = word_tokenize(text) #Tokenize the text into individual words
    
    tokens = [word for word in tokens if word.isalnum()] #Remove special characters and keep only alphanumeric tokens

    stop_words = set(stopwords.words("english"))#Remove stop words (common words like 'a', 'the', 'is')
    tokens = [word for word in tokens if word not in stop_words]

    ps = PorterStemmer() #Apply stemming (reducing words to their root form)
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")
    cleaned = preprocess_text(message)
    
    # Transform the cleaned text into numerical features using the loaded tfidf vectorizer
    # The input must be a list of strings: [cleaned]
    vectorized = tfidf_vectorizer.transform([cleaned])
    
    #Make the prediction using the Random Forest model rf_model
    prediction = rf_model.predict(vectorized)[0]
    
    #Convert the numerical prediction (ham 0 or spam 1)
    label = "spam" if prediction == 1 else "ham"
    
    #return prediction result as a json response
    return jsonify({"prediction": label})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
