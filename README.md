# 📧 SMS Spam Classification Project
Mustafa Al-Bayati & Daniel Jönsson

Kristianstad University

This project uses **Machine Learning** to classify text messages as **Spam** or **Ham (Not Spam)**. 
There is two files in this project the first one is **main.ipynb** we followed the course requirements. 
Another file is **spam-classifier.ipynb** we did it as end-to-end project includes a **Backend** built with **Flask** and a simple **Frontend** built with **Streamlit**.

Both files tested with different algorithms. 

**Some info** down about the end-to-end project:
## Model Performance (Logistic Regression from Scratch)

After training and evaluating the Logistic Regression model that we implemented from scratch, the model achieved the following results:

_ Logistic Regression (Scratch) Results _
Accuracy: 0.9622823984526112

Classification Report:

               precision    recall  f1-score   support

           0       0.96      1.00      0.98       916
           1       0.96      0.69      0.81       118

    accuracy                           0.96      1034
   macro avg       0.96      0.85      0.89      1034
weighted avg       0.96      0.96      0.96      1034


## Summary:

The model performed very well, reaching an overall accuracy of 96%.
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

The backend (spam_classify.py) loads a trained Logistic Regression from Scratch.

The frontend (streamlit_app.py) allows users to enter messages and get predictions in real-time.

The model was trained on the SMS Spam Collection Dataset, which contains labeled examples of spam and ham messages.


**Example Predictions**

“You’ve been selected for a free offer.”“Win $500 cash now!” --> prediction  -->  Spam

Hi mom, how are you? Let’s meet for lunch tomorrow. --> prediction --> ham

