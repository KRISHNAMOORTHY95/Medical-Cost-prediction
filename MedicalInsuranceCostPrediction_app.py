
# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# Configure page
st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="🏥", layout="wide")

# Load or create model
@st.cache_data
def load_or_create_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning("Model not found. Creating a new one with synthetic data.")
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'age': np.random.randint(18, 65, n),
            'bmi': np.clip(np.random.normal(28, 6, n), 15, 50),
            'children': np.random.randint(0, 5, n),
            'sex': np.random.choice([0, 1], n),
            'smoker': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'region': np.random.choice([0, 1, 2, 3], n)
        })
        df['charges'] = (df['age'] * 200 + df['bmi'] * 100 + df['children'] * 500 +
                         df['smoker'] * 15000 + df['sex'] * 200 + df['region'] * 300 +
                         np.random.normal(0, 2000, n))
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(df.drop('charges', axis=1), df['charges'])
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        return model

# Load or create dataset
@st.cache_data
def load_or_create_dataset():
    try:
        df = pd.read_csv("medical_insurance.csv")
        return df
    except FileNotFoundError:
        st.warning("Dataset not found. Using synthetic data.")
        return pd.DataFrame({
            'age': np.random.randint(18, 65, 1000),
            'bmi': np.clip(np.random.normal(28, 6, 1000), 15, 50),
            'children': np.random.randint(0, 5, 1000),
            'sex': np.random.choice([0, 1], 1000),
            'smoker': np.random.choice([0, 1], 1000, p=[0.8, 0.2]),
            'region': np.random.choice([0, 1, 2, 3], 1000),
            'charges': np.random.uniform(2000, 40000, 1000)
        })

model = load_or_create_model()
df = load_or_create_dataset()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Visualizations", "Predict Cost"])

if page == "Introduction":
    st.title("🏥 Medical Insurance Cost Prediction")
    st.write("This app predicts insurance cost using a machine learning model trained on synthetic or provided data.")

elif page == "Visualizations":
    st.title("📊 Visual Explorations")
    plot_type = st.selectbox("Choose a visualization:", [
        "Distribution of Charges", "Charges by Gender", "Charges vs Age"
    ])

    if plot_type == "Distribution of Charges":
        fig, ax = plt.subplots()
        sns.histplot(df['charges'], kde=True, ax=ax)
        ax.set_title("Distribution of Insurance Charges")
        st.pyplot(fig)

    elif plot_type == "Charges by Gender":
        fig, ax = plt.subplots()
        df['sex_label'] = df['sex'].map({0: 'Female', 1: 'Male'})
        sns.boxplot(x='sex_label', y='charges', data=df, ax=ax)
        ax.set_title("Charges by Gender")
        st.pyplot(fig)

    elif plot_type == "Charges vs Age":
        fig, ax = plt.subplots()
        sns.scatterplot(x='age', y='charges', hue=df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'}), data=df, ax=ax)
        ax.set_title("Charges vs Age (Smoker Status)")
        st.pyplot(fig)

elif page == "Predict Cost":
    st.title("💰 Predict Insurance Cost")
    with st.form("predict_form"):
        age = st.slider("Age", 18, 100, 30)
        bmi = st.slider("BMI", 10.0, 60.0, 25.0)
        children = st.slider("Children", 0, 5, 0)
        sex = st.radio("Sex", ["Female", "Male"])
        smoker = st.radio("Smoker", ["No", "Yes"])
        region = st.selectbox("Region", ["Northeast", "Southeast", "Southwest", "Northwest"])
        submit = st.form_submit_button("Predict")

    if submit:
        sex = 1 if sex == "Male" else 0
        smoker = 1 if smoker == "Yes" else 0
        region_map = {"Northeast": 0, "Southeast": 1, "Southwest": 2, "Northwest": 3}
        region = region_map[region]
        input_data = pd.DataFrame([[age, bmi, children, sex, smoker, region]],
                                  columns=['age', 'bmi', 'children', 'sex', 'smoker', 'region'])
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Insurance Charge: ${prediction:,.2f}")
