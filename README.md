# Disease Prediction Web App

This project is a Flask-based web application for predicting common diseases using pre-trained machine learning models. The app provides a simple web interface to input patient data and returns predictions for diseases such as heart disease, diabetes, and breast cancer.

## Features

- Predict heart disease, diabetes, and breast cancer using saved models.
- Simple web UI built with Flask and HTML templates.
- Models are stored in the `model/` directory and loaded at runtime.

## Repository structure

- `app.py` - Main Flask application entry point.
- `app1.py` - Alternate or testing Flask app (if present).
- `model/` - Pre-trained model files (e.g., `.pkl`, `.h5`).
- `templates/` - HTML templates used by the app.

## Setup

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2. Install dependencies (Flask, scikit-learn, tensorflow/keras as needed):

```bash
pip install -r requirements.txt
```

If there is no `requirements.txt`, install the common packages:

```bash
pip install flask numpy pandas scikit-learn tensorflow
```

## Running the app

1. Ensure the `model/` directory contains the pre-trained models (examples included in this repo).
2. Start the Flask app:

```bash
python app.py
```

3. Open a browser and go to `http://127.0.0.1:5000/`.

## Usage

- Use the web form to enter required patient metrics (age, sex, blood pressure, glucose, etc.).
- Submit the form to receive a prediction and a short explanation.

## Models included

- `heart_disease_model.pkl` - Scikit-learn model for heart disease prediction.
- `diabitse_model.h5` - Keras model for diabetes prediction.
- `breast_cancer_model.pkl` - Scikit-learn model for breast cancer prediction.

> Note: File names are listed as they appear in the `model/` directory. If you replace model files, update the code accordingly.

## Notes and next steps

- Add `requirements.txt` with exact versions used for training for reproducibility.
- Add unit tests for model inference functions.
- Improve input validation and add user-friendly error messages.

## License

This project is provided as-is. Add a license file if you plan to share the project publicly. 