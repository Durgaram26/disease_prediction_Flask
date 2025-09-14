from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
def load_models():
    # Breast Cancer Model (pickle)
    with open(os.path.join("model", "breast_cancer_model.pkl"), "rb") as f:
        bc_model = pickle.load(f)
    
    # Diabetes Model (Keras H5)
    diabetes_model = load_model(os.path.join("model", "diabitse_model.h5"))
    
    # Parkinson's Model (pickle) and its scaler
    with open(os.path.join("model", "parkinsondisease_model.pkl"), "rb") as f:
        parkinson_model = pickle.load(f)
    with open(os.path.join("model", "parksscaler_file.pkl"), "rb") as f:
        parksscaler = pickle.load(f)
    
    # Heart Disease Model (pickle)
    with open(os.path.join("model", "heart_disease_model.pkl"), "rb") as f:
        heart_model = pickle.load(f)
    
    # Kidney Disease Model (pickle)
    with open(os.path.join("model", "kidney_model.pkl"), "rb") as f:
        kidney_model = pickle.load(f)
    
    return (bc_model, diabetes_model, parkinson_model, heart_model, kidney_model, parksscaler)

models = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    disease_type = data.get('disease')
    features = data.get('features')
    
    if disease_type == "breast_cancer":
        bc_model = models[0]
        features_array = np.array([features])
        prediction = bc_model.predict(features_array)
        result = "Malignant" if prediction[0] == 4 else "Benign"
        
    elif disease_type == "diabetes":
        diabetes_model = models[1]
        features_array = np.array([features])
        prediction = diabetes_model.predict(features_array)
        result = "Diabetic" if prediction[0] >= 0.5 else "Non-Diabetic"
        
    elif disease_type == "parkinsons":
        parkinson_model = models[2]
        parksscaler = models[5]
        features_array = np.array([features])
        features_scaled = parksscaler.transform(features_array)
        prediction = parkinson_model.predict(features_scaled)
        result = "Parkinson's Detected" if prediction[0] == 1 else "No Parkinson's Detected"
        
    elif disease_type == "heart":
        heart_model = models[3]
        features_array = np.array([features])
        prediction = heart_model.predict(features_array)
        result = "Heart Disease Present" if prediction[0] == 1 else "No Heart Disease Detected"
        
    elif disease_type == "kidney":
        kidney_model = models[4]
        features_array = np.array([features])
        prediction = kidney_model.predict(features_array)
        result = "Kidney Disease Detected" if prediction[0] == 1 else "No Kidney Disease Detected"
    
    else:
        result = "Invalid disease type"
    
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
