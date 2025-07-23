from flask import Blueprint, render_template, request
from flask_login import login_required, current_user
import joblib
import os
from website.model import Url_features
from transformers import pipeline

views = Blueprint('views', __name__)

# Load URL phishing detection model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'Random Forest_model.pkl')
mlmodel = joblib.load(model_path)

# Load Hugging Face phishing text detection model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@views.route('/')
@login_required
def home():
    return render_template("home.html", user=current_user)

@views.route('/predict_type')
def predict_options():
    return render_template('predict_type.html')

@views.route('/detect-text', methods=['GET', 'POST'])
def detect_text():
    if request.method == 'POST':
        text = request.form.get("textInput")
        
        # AI Model Classification
        labels = ["phishing", "safe"]
        result = classifier(text, labels)
        prediction = result['labels'][0]  # Most probable label
        confidence = round(result['scores'][0] * 100, 2)  # Confidence score
        
        return render_template('text_result.html', result=prediction, confidence=confidence)
    
    return render_template('detect_text.html')

@views.route('/predict', methods=['GET', 'POST'])
def predicts():
    if request.method == 'GET':
        return render_template('predict.html')
    
    if request.method == 'POST':
        url = request.form.get("url")
        features = Url_features.extract_features(url)
        prediction = mlmodel.predict([features])
        prediction_proba = mlmodel.predict_proba([features])[0][1]  # Probability of being phishing
        
        result = 'Phishing' if prediction[0] == 1 else 'Legitimate'
        confidence_percentage = round(prediction_proba * 100, 2)
        
        return render_template('result.html', result=result, confidence=confidence_percentage)
