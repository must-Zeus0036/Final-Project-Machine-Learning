# 📧 SMS Spam Classification Project
Mustafa Al-Bayati & Daniel Jönsson

Kristianstad University

This project uses **Machine Learning** to classify text messages as **Spam** or **Ham (Not Spam)**.  
It includes a backend built with **Flask** and a simple frontend built with **Streamlit**.


## Model Performance (Logistic Regression from Scratch)

After training and evaluating the Logistic Regression model that I implemented from scratch, the model achieved the following results:

Metric	Score
----------------
## Accuracy	0.962
## Precision (Spam = 1)	0.96
## Recall (Spam = 1)	0.69
## F1-Score (Spam = 1)	0.81

## Summary:

The model performed very well, reaching an overall accuracy of 96.2%.
It can correctly identify most spam messages with high precision and balanced recall.
These results confirm that a Logistic Regression model built completely from scratch using NumPy can achieve strong and reliable performance for spam detection tasks.

---

## Getting Started

Follow these steps to clone the repository and run it locally.

### 1. Clone the Repository
Open your terminal (or Git Bash) and run:
```bash
git clone https://github.com/must-Zeus0036/machine_learning_project.git
```
 
Then move into the project folder:
```
cd machine_learning_project
```

## Open in VS Code
```
code .
```

## Create and Activate a Virtual Environment
```
python -m venv .venv
```

Activate it:

On Windows:
```
.venv\Scripts\activate
```

On macOS/Linux:
```
source .venv/bin/activate
```
Install the required Python packages:
```
pip install -r requirements.txt
```

If the requirements.txt file is missing, you can manually install them:
```
pip install flask streamlit scikit-learn pandas nltk matplotlib seaborn
```

Run the Flask Backend

Start the backend server by running:
```
python spam_classify.py
```

You should see something like:
```
Running on http://127.0.0.1:5000
```

⚠️ Keep this terminal running — it hosts your Flask backend API.

Run the Streamlit Frontend

Open a new terminal tab or window and run:
```
streamlit run streamlit_app.py
```

Then open your browser and visit:
```
http://localhost:8501
```


## How It Works

The backend (spam_classify.py) loads a trained Random Forest model and TF-IDF vectorizer.

The frontend (streamlit_app.py) allows users to enter messages and get predictions in real-time.

The model was trained on the SMS Spam Collection Dataset, which contains labeled examples of spam and ham messages.


🧩 Example Predictions

“You’ve been selected for a free offer.”“Win $500 cash now!” --> prediction  -->  Spam

Hi mom, how are you? Let’s meet for lunch tomorrow. --> prediction --> ham

