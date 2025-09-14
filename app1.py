import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

st.title("Multi-Disease Prediction System")

# Load models
@st.cache_resource
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

(bc_model, diabetes_model, parkinson_model, heart_model, kidney_model, parksscaler) = load_models()

# Disease selection
disease = st.selectbox("Select Disease to Predict", 
                       ["Breast Cancer", "Diabetes", "Parkinson's", "Heart Disease", "Kidney Disease"])

if disease == "Breast Cancer":
    st.header("Breast Cancer Prediction")
    st.markdown("**Enter the following attributes (values between 1 and 10):**")
    clump_thickness = st.number_input("Clump Thickness", min_value=1, max_value=10, value=5)
    cell_size = st.number_input("Uniformity of Cell Size", min_value=1, max_value=10, value=5)
    cell_shape = st.number_input("Uniformity of Cell Shape", min_value=1, max_value=10, value=5)
    marginal_adhesion = st.number_input("Marginal Adhesion", min_value=1, max_value=10, value=5)
    epithelial_cell_size = st.number_input("Single Epithelial Cell Size", min_value=1, max_value=10, value=5)
    bare_nuclei = st.number_input("Bare Nuclei", min_value=1, max_value=10, value=5)
    bland_chromatin = st.number_input("Bland Chromatin", min_value=1, max_value=10, value=5)
    normal_nucleoli = st.number_input("Normal Nucleoli", min_value=1, max_value=10, value=5)
    mitoses = st.number_input("Mitoses", min_value=1, max_value=10, value=5)

    if st.button("Predict Breast Cancer", key="btn_breast"):
        features = np.array([[clump_thickness, cell_size, cell_shape, marginal_adhesion,
                              epithelial_cell_size, bare_nuclei, bland_chromatin, normal_nucleoli, mitoses]])
        prediction = bc_model.predict(features)
        result = "Malignant" if prediction[0] == 4 else "Benign"
        st.success(f"Breast Cancer Prediction: {result}")

elif disease == "Diabetes":
    st.header("Diabetes Prediction")
    st.markdown("**Enter the following attributes:**")
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, value=2)
    glucose = st.number_input("Plasma Glucose Concentration", min_value=0, value=100)
    bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0, value=70)
    skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0, value=20)
    insulin = st.number_input("2-Hour Serum Insulin", min_value=0, value=80)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, value=25.0, format="%.2f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=0, value=30)

    if st.button("Predict Diabetes", key="btn_diabetes"):
        features = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(features)
        # Assuming the model returns probabilities or binary predictions
        result = "Diabetic" if prediction[0] >= 0.5 else "Non-Diabetic"
        st.success(f"Diabetes Prediction: {result}")

elif disease == "Parkinson's":
    st.header("Parkinson's Disease Prediction")
    st.markdown("**Enter the following attributes:**")
    
    mdvpFo = st.number_input("MDVP Fo (Hz)", value=120.0, format="%.2f")
    mdvpFhi = st.number_input("MDVP Fhi (Hz)", value=150.0, format="%.2f")
    mdvpFlo = st.number_input("MDVP Flo (Hz)", value=80.0, format="%.2f")
    mdvpJitterPercent = st.number_input("MDVP Jitter Percent", value=0.01, format="%.4f")
    mdvpJitterAbsolute = st.number_input("MDVP Jitter Absolute", value=0.002, format="%.4f")
    mdvpRap = st.number_input("MDVP Rap", value=0.3, format="%.2f")
    mdvpPPQ = st.number_input("MDVP PPQ", value=0.3, format="%.2f")
    jitterDDP = st.number_input("Jitter DDP", value=0.1, format="%.2f")
    mdvpJimmer = st.number_input("MDVP Jimmer", value=0.0, format="%.2f")
    mdvpJimmerDB = st.number_input("MDVP Jimmer DB", value=0.0, format="%.2f")
    shimmerAPQ3 = st.number_input("Shimmer APQ3", value=0.1, format="%.2f")
    shimmerAPQ5 = st.number_input("Shimmer APQ5", value=0.1, format="%.2f")
    mdvpAPQ = st.number_input("MDVP APQ", value=0.1, format="%.2f")
    shimmerDDA = st.number_input("Shimmer DDA", value=0.1, format="%.2f")
    NHR = st.number_input("NHR", value=0.1, format="%.2f")
    HNR = st.number_input("HNR", value=20.0, format="%.2f")
    RPDE = st.number_input("RPDE", value=0.5, format="%.2f")
    DFA = st.number_input("DFA", value=0.5, format="%.2f")
    spread1 = st.number_input("Spread1", value=0.1, format="%.2f")
    spread2 = st.number_input("Spread2", value=0.1, format="%.2f")
    d2 = st.number_input("D2", value=2.0, format="%.2f")
    PPE = st.number_input("PPE", value=0.1, format="%.2f")
    
    if st.button("Predict Parkinson's", key="btn_parkinson"):
         # Arrange features in the order used during training (22 features)
         features = np.array([[mdvpFo, mdvpFhi, mdvpFlo, mdvpJitterPercent, mdvpJitterAbsolute,
                               mdvpRap, mdvpPPQ, jitterDDP, mdvpJimmer, mdvpJimmerDB, shimmerAPQ3,
                               shimmerAPQ5, mdvpAPQ, shimmerDDA, NHR, HNR, RPDE, DFA, spread1, spread2, d2, PPE]])
         # Scale the features using the loaded scaler
         features_scaled = parksscaler.transform(features)
         prediction = parkinson_model.predict(features_scaled)
         result = "Parkinson's Detected" if prediction[0] == 1 else "No Parkinson's Detected"
         st.success(result)

elif disease == "Heart Disease":
    st.header("Heart Disease Prediction")
    st.markdown("**Enter the following attributes:**")
    
    age_hd = st.number_input("Age", min_value=0, value=52)
    sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0], index=1)[1]
    cp = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=3, value=0)
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, value=125)
    chol = st.number_input("Cholesterol (chol)", min_value=0, value=212)
    fbs = st.selectbox("Fasting Blood Sugar (fbs)", options=[("False", 0), ("True", 1)], format_func=lambda x: x[0], index=0)[1]
    restecg = st.number_input("Resting ECG (restecg)", min_value=0, max_value=2, value=1)
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=0, value=168)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], index=0)[1]
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, value=1.0, format="%.1f")
    slope = st.number_input("Slope of Peak Exercise ST Segment (slope)", min_value=0, max_value=2, value=2)
    ca = st.number_input("Number of Major Vessels (ca)", min_value=0, max_value=3, value=2)
    thal = st.selectbox("Thal", options=[3, 6, 7], index=0)

    if st.button("Predict Heart Disease", key="btn_heart_updated"):
         features = np.array([[age_hd, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                               oldpeak, slope, ca, thal]])
         prediction = heart_model.predict(features)
         result = "Heart Disease Present" if prediction[0] == 1 else "No Heart Disease Detected"
         st.success(f"Heart Disease Prediction: {result}")

elif disease == "Kidney Disease":
    st.header("Kidney Disease Prediction")
    st.markdown("**Enter the following attributes:**")
    age_k = st.number_input("Age", min_value=0, value=23)
    bp_k = st.number_input("Blood Pressure", min_value=0, value=60)
    sg_k = st.selectbox("Specific Gravity", options=[1.005, 1.010, 1.015, 1.020, 1.025], index=3)
    al_k = st.number_input("Albumin (0-5)", min_value=0, max_value=5, value=0)
    su_k = st.number_input("Sugar (0-5)", min_value=0, max_value=5, value=0)
    rbc = st.selectbox("Red Blood Cells", options=["normal", "abnormal"], index=0)
    pc = st.selectbox("Pus Cell", options=["normal", "abnormal"], index=0)
    pcc = st.selectbox("Pus Cell Clumps", options=["present", "notpresent"], index=1)
    ba = st.selectbox("Bacteria", options=["present", "notpresent"], index=1)
    bgr_k = st.number_input("Blood Glucose Random (mg/dl)", min_value=0, value=95)
    bu_k = st.number_input("Blood Urea (mg/dl)", min_value=0, value=24)
    sc_k = st.number_input("Serum Creatinine (mg/dl)", min_value=0.0, value=0.8, format="%.2f")
    sod_k = st.number_input("Sodium (mEq/L)", min_value=0, value=145)
    pot_k = st.number_input("Potassium (mEq/L)", min_value=0, value=5)
    hb = st.number_input("Haemoglobin (gms)", min_value=0.0, value=15.0, format="%.2f")
    pcv = st.number_input("Packed Cell Volume", min_value=0, value=52)
    wbc = st.number_input("White Blood Cell Count", min_value=0, value=6300)
    rbc_count = st.number_input("Red Blood Cell Count", min_value=0.0, value=4.6, format="%.2f")
    htn = st.selectbox("Hypertension", options=[("no", 0), ("yes", 1)], format_func=lambda x: x[0])[1]
    dm = st.selectbox("Diabetes Mellitus", options=[("no", 0), ("yes", 1)], format_func=lambda x: x[0])[1]
    cad = st.selectbox("Coronary Artery Disease", options=[("no", 0), ("yes", 1)], format_func=lambda x: x[0])[1]
    appetite = st.selectbox("Appetite", options=["good", "poor"], index=0)
    peda_edema = st.selectbox("Peda Edema", options=[("no", 0), ("yes", 1)], format_func=lambda x: x[0])[1]
    aanemia = st.selectbox("Aanemia", options=[("no", 0), ("yes", 1)], format_func=lambda x: x[0])[1]

    if st.button("Predict Kidney Disease", key="btn_kidney"):
        # Convert categorical variables to numerical values.
        rbc_val = 1 if rbc == "normal" else 0
        pc_val = 1 if pc == "normal" else 0
        pcc_val = 1 if pcc == "present" else 0
        ba_val = 1 if ba == "present" else 0
        appetite_val = 1 if appetite == "good" else 0
        
        # Arrange features in the same order as during training.
        features = np.array([[age_k, bp_k, sg_k, al_k, su_k, rbc_val, pc_val, pcc_val, ba_val,
                              bgr_k, bu_k, sc_k, sod_k, pot_k, hb, pcv, wbc, rbc_count,
                              htn, dm, cad, appetite_val, peda_edema, aanemia]])
        prediction = kidney_model.predict(features)
        result = "Kidney Disease Detected" if prediction[0] == 1 else "No Kidney Disease Detected"
        st.success(result)
