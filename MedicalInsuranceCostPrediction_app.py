# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="🏥", layout="wide")

# Load or train model
@st.cache_resource
def load_or_create_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        n = 1000
        df = pd.DataFrame({
            "age": np.random.randint(18, 65, n),
            "bmi": np.random.normal(28, 6, n).clip(15, 50),
            "children": np.random.randint(0, 5, n),
            "sex": np.random.randint(0, 2, n),
            "smoker": np.random.randint(0, 2, n),
            "region": np.random.randint(0, 4, n),
        })
        df["charges"] = df["age"] * 200 + df["bmi"] * 100 + df["children"] * 500 + \
                        df["smoker"] * 15000 + df["sex"] * 200 + df["region"] * 300 + \
                        np.random.normal(0, 2000, n)
        df["charges"] = df["charges"].clip(1000, 50000)
        X = df[["age", "bmi", "children", "sex", "smoker", "region"]]
        y = df["charges"]
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        return model

# Load data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("medical_insurance.csv")
    except:
        return pd.DataFrame({
            "age": np.random.randint(18, 65, 1000),
            "bmi": np.random.normal(28, 6, 1000).clip(15, 50),
            "children": np.random.randint(0, 5, 1000),
            "sex": np.random.randint(0, 2, 1000),
            "smoker": np.random.randint(0, 2, 1000),
            "region": np.random.randint(0, 4, 1000),
            "charges": np.random.randint(2000, 30000, 1000)
        })

model = load_or_create_model()
df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualization", "Prediction"])

if page == "Home":
    st.title("🏥 Medical Insurance Cost Prediction")
    st.markdown("""
    Predict medical insurance charges based on:
    - Age
    - BMI
    - Gender
    - Children
    - Smoker
    - Region
    """)
    st.metric("Total records", len(df))

elif page == "Visualization":
    st.title("📊 Exploratory Data Analysis")
    chart = st.selectbox("Choose visualization", [
        "Distribution of Charges",
        "Charges vs Age",
        "Charges vs BMI",
        "Correlation Matrix"
    ])
    if chart == "Distribution of Charges":
        fig, ax = plt.subplots()
        ax.hist(df["charges"], bins=30, color="teal", edgecolor="black")
        st.pyplot(fig)
elif chart == "Charges vs Age":
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="age", y="charges", hue="smoker", ax=ax)
    ax.set_title("Charges vs Age")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

elif chart == "Charges vs BMI":
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", ax=ax)
    ax.set_title("Charges vs BMI")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

elif chart == "Correlation Matrix":
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

elif page == "Prediction":
    st.title("💰 Insurance Charge Prediction")
    with st.form("predict_form"):
        age = st.slider("Age", 18, 100, 30)
        bmi = st.slider("BMI", 15.0, 50.0, 28.0)
        children = st.slider("Children", 0, 5, 0)
        sex = st.radio("Gender", ["Female", "Male"])
        smoker = st.radio("Smoker", ["No", "Yes"])
        region = st.selectbox("Region", ["Northeast", "Southeast", "Southwest", "Northwest"])
        submitted = st.form_submit_button("Predict")
    if submitted:
        input_df = pd.DataFrame({
            "age": [age],
            "bmi": [bmi],
            "children": [children],
            "sex": [1 if sex=="Male" else 0],
            "smoker": [1 if smoker=="Yes" else 0],
            "region": [ {"Northeast":0, "Southeast":1, "Southwest":2, "Northwest":3}[region] ]
        })
        pred = model.predict(input_df)[0]
        st.success(f"Estimated Insurance Charges: ${pred:,.2f}")
