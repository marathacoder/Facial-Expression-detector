# Import necessary libraries
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json
import flask
from flask import request, jsonify

# Load pre-trained NLP model
nlp = spacy.load("en_core_web_sm")

# Define function to extract features from CV text
def extract_features(cv_text):
    doc = nlp(cv_text)
    features = {
        "education": [],
        "experience": [],
        "skills": [],
        "achievements": []
    }
    # Extract educational background
    for ent in doc.ents:
        if ent.label_ == "ORG":
            features["education"].append(ent.text)
        if ent.label_ == "DATE":
            features["experience"].append(ent.text)
    # Extract skills and achievements (custom logic needed here)
    # ...
    return features

# Define function to preprocess and vectorize text
def preprocess_and_vectorize(cvs):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(cvs)
    return vectors, vectorizer

# Load and preprocess data
cv_data = pd.read_csv("C:/Users/Lenovo/Downloads/Yash-Kute-B.Tech.-ArtificialIntelligence_DataScience-2024-03-11-02-54-46-255307.csv")  # Replace with actual dataset
cvs = cv_data["cv_text"].apply(extract_features)
vectors, vectorizer = preprocess_and_vectorize(cvs)

# Define labels (this example assumes labeled data)
labels = cv_data["personality_traits"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Create Flask API for deployment
app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/predict', methods=['POST'])
def predict():
    cv_text = request.form['cv']
    features = extract_features(cv_text)
    vector = vectorizer.transform([features])
    prediction = model.predict(vector)
    return jsonify({"personality_traits": prediction[0]})

app.run()
