import pickle
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score
import pandas as pd

# Set page title, layout, and favicon
st.set_page_config(page_title="🌿 Diabetes Prediction", layout="wide", page_icon="🌿")

# Load the trained model
diabetes_model_path = r"C:\Users\desai\OneDrive\Desktop\disease prediction\diabetes_model.sav"

try:
    with open(diabetes_model_path, "rb") as model_file:
        diabetes_model = pickle.load(model_file)
except Exception as e:
    st.error(f"⚠️ Error loading the model: {e}")

# Custom CSS for a light pastel relaxing theme
st.markdown("""
    <style>
    body {
        background-color: #EAF6F6;
        color: #3A506B;
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #CDEDF6, #EAF6F6, #D4E6F1);
    }
    .stButton>button {
        background-color: #A0C4FF;
        color: white;
        border-radius: 15px;
        padding: 12px;
        width: 100%;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 8px rgba(160, 196, 255, 0.4);
    }
    .stButton>button:hover {
        background-color: #83A9FF;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 10px;
        border: 2px solid #CDEDF6;
        background-color: #F0F8FF;
        color: #3A506B;
    }
    .prediction-box {
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 4px 8px rgba(120, 180, 210, 0.3);
    }
    .header {
        text-align: center;
        color: #5A7684;
        font-weight: bold;
    }
    .accuracy-box {
        background-color: #E3F2FD;
        color: #3A506B;
        font-size: 16px;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='header'>🌿 Diabetes Prediction using ML 🌿</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #5A7684;'>Enter your details below to check for diabetes risk.</h3>", unsafe_allow_html=True)

# Load dataset for real-time accuracy calculation (Replace with actual dataset path)
dataset_path = r"C:\Users\desai\OneDrive\Desktop\disease prediction\diabetes.csv"

try:
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=["Outcome"])  # Features
    y = data["Outcome"]  # Target (0: Non-Diabetic, 1: Diabetic)

    # Calculate model accuracy on the entire dataset
    y_pred = diabetes_model.predict(X)
    real_time_accuracy = accuracy_score(y, y_pred)

    # Input form using Streamlit columns
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("🤰 Number of Pregnancies", "0")

    with col2:
        Glucose = st.text_input("🍬 Glucose Level", "0")

    with col3:
        BloodPressure = st.text_input("💓 Blood Pressure", "0")

    with col1:
        SkinThickness = st.text_input("📏 Skin Thickness", "0")

    with col2:
        Insuline = st.text_input("💉 Insulin Level", "0")

    with col3:
        BMI = st.text_input("⚖️ BMI Value", "0.0")

    with col1:
        DiabetesPedigreeFunction = st.text_input("🧬 Diabetes Pedigree Function", "0.0")

    with col2:
        Age = st.text_input("🎂 Age", "0")

    # Prediction Button
    if st.button("🌿 Predict Diabetes"):
        try:
            # Convert user inputs safely
            user_input = np.array([
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insuline), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]).reshape(1, -1)

            # Predict diabetes
            diab_prediction = diabetes_model.predict(user_input)

            st.markdown(f"<div class='accuracy-box'>📊 Real-Time Model Accuracy: <b>{real_time_accuracy*100:.2f}%</b></div>", unsafe_allow_html=True)

            # Display result with a soft pastel theme
            if diab_prediction[0] == 1:
                st.markdown("<div class='prediction-box' style='background-color: #A0C4FF; color: white;'>❌ The person is DIABETIC.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='prediction-box' style='background-color: #98FB98; color: #3A506B;'>✅ The person is NOT DIABETIC.</div>", unsafe_allow_html=True)

        except ValueError:
            st.error("⚠️ Please enter valid numerical values.")

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

except Exception as e:
    st.error(f"⚠️ Error loading dataset: {e}")
