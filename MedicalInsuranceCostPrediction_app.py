import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(page_title="Medical Insurance Cost Prediction", layout="wide")

# Load the trained model
model = joblib.load("models/best_model.pkl")

# Load and preprocess dataset (you can customize this function as needed)
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("data/medical_insurance.csv")
    # Encode categorical columns for visualization (not for modeling)
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df['region'] = df['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})
    return df

st.title("💰 Medical Insurance Cost Estimator")

# Tab Layout
tab1, tab2 = st.tabs(["📊 EDA Insights", "💡 Predict Insurance Cost"])

# Tab 1: EDA
with tab1:
    st.header("📊 Exploratory Data Analysis")

    df = load_and_preprocess()

    st.subheader("Distribution of Medical Charges")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['charges'], kde=True, ax=ax1, color="skyblue")
    st.pyplot(fig1)

    st.subheader("Charges vs Age by Smoking Status")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='age', y='charges', hue='smoker', alpha=0.6, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", ax=ax3)
    st.pyplot(fig3)

# Tab 2: Prediction
with tab2:
    st.header("💡 Predict Your Insurance Cost")

    age = st.slider("Age", 18, 65, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.slider("Number of Children", 0, 5, 0)
    smoker = st.selectbox("Do you Smoke?", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    # Encode user inputs
    sex_encoded = 1 if sex == "male" else 0
    smoker_encoded = 1 if smoker == "yes" else 0
    region_encoded = {"northeast": 3, "northwest": 2, "southeast": 1, "southwest": 0}[region]

    input_data = pd.DataFrame([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]],
                              columns=["age", "sex", "bmi", "children", "smoker", "region"])

    if st.button("Predict Insurance Cost"):
        prediction = model.predict(input_data)[0]
        st.success(f"💲 Estimated Insurance Charges: ${prediction:,.2f}")
