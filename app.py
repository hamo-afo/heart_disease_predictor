"""
Heart Disease Prediction System - Streamlit app
Loads svc_trained_model.pkl made by Heart_Disease_Prediction.ipynb.
The LabelEncoders are rebuilt from the same fixed category lists used in
the notebook, so the numeric codes (and predictions) are identical.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

CATEGORIES = {
    "ChestPain": ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    "ExerciseAngina": ["No", "Yes"],
    "STSlope": ["Upsloping", "Flat", "Downsloping"],
    "Thal": ["Normal", "Fixed Defect", "Reversable Defect"],
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    return {name: LabelEncoder().fit(values) for name, values in CATEGORIES.items()}


model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    "Choose the four values below. The trained Support Vector Classifier "
    "will predict if heart disease is likely.\n\n"
    "*Test accuracy: about 79% (UCI Cleveland data, 301 patients).*"
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    chest_pain = st.selectbox("Chest Pain Type", CATEGORIES["ChestPain"], index=3)
    exercise_angina = st.selectbox("Exercise Induced Angina", CATEGORIES["ExerciseAngina"])
with col2:
    st_slope = st.selectbox("ST Slope (peak exercise)", CATEGORIES["STSlope"], index=1)
    thal = st.selectbox("Thal (blood test)", CATEGORIES["Thal"])

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "STSlope": [st_slope],
        "Thal": [thal],
    })
    encoded = pd.DataFrame({c: encoders[c].transform(user_input[c]) for c in CATEGORIES})
    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE LIKELY")
    else:
        st.success("### ✅ NO HEART DISEASE LIKELY")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption("Learning project only. This is not medical advice and must not be used for diagnosis.")
