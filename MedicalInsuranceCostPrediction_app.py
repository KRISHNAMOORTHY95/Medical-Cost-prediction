
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

# Page config
st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="🏥", layout="wide")

# Load or create model
@st.cache_data
def load_or_create_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning("Model file not found. Creating a new model.")
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'age': np.random.randint(18, 65, n_samples),
            'bmi': np.clip(np.random.normal(28, 6, n_samples), 15, 50),
            'children': np.random.randint(0, 6, n_samples),
            'smoker': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'sex': np.random.choice([0, 1], n_samples),
            'region': np.random.choice([0, 1, 2, 3], n_samples)
        })
        df['charges'] = (
            df['age'] * 200 + df['bmi'] * 100 + df['children'] * 500 +
            df['smoker'] * 15000 + df['sex'] * 200 + df['region'] * 300 +
            np.random.normal(0, 2000, n_samples)
        )
        df['charges'] = np.clip(df['charges'], 1000, 50000)
        X = df[['age', 'bmi', 'children', 'sex', 'smoker', 'region']]
        y = df['charges']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        return model

# Load or create dataset
@st.cache_data
def load_or_create_dataset():
    try:
        return pd.read_csv("medical_insurance.csv")
    except FileNotFoundError:
        st.warning("Dataset not found. Using sample data.")
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'age': np.random.randint(18, 65, n_samples),
            'bmi': np.clip(np.random.normal(28, 6, n_samples), 15, 50),
            'children': np.random.randint(0, 6, n_samples),
            'smoker': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'sex': np.random.choice([0, 1], n_samples),
            'region': np.random.choice([0, 1, 2, 3], n_samples)
        })
        df['charges'] = (
            df['age'] * 200 + df['bmi'] * 100 + df['children'] * 500 +
            df['smoker'] * 15000 + df['sex'] * 200 + df['region'] * 300 +
            np.random.normal(0, 2000, n_samples)
        )
        df['charges'] = np.clip(df['charges'], 1000, 50000)
        return df

model = load_or_create_model()
df = load_or_create_dataset()

# Sidebar Navigation
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Introduction", "📊 Visualizations", "💰 Prediction"])

# Introduction Page
if page == "🏠 Introduction":
    st.title("🏥 Medical Insurance Cost Prediction")
    st.markdown("""
    This Streamlit app uses a Random Forest model to predict medical insurance charges
    based on age, BMI, gender, smoking status, number of children, and region.
    """)
    st.info(f"**Total Records:** {len(df):,}  
**Average Charge:** ${df['charges'].mean():,.2f}")

# Visualizations Page
elif page == "📊 Visualizations":
    st.title("📊 Data Visualizations")
    df['sex_label'] = df['sex'].map({0: 'Female', 1: 'Male'})
    df['smoker_label'] = df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
    df['region_label'] = df['region'].map({0: 'Northeast', 1: 'Southeast', 2: 'Southwest', 3: 'Northwest'})

    viz = st.selectbox("Choose Visualization", [
        "Distribution of Charges", "Charges by Gender", "Charges by Smoking Status",
        "Charges vs BMI", "Charges vs Age", "Region-wise Policyholders", "Children vs Charges"
    ])

    fig, ax = plt.subplots(figsize=(10, 6))
    if viz == "Distribution of Charges":
        sns.histplot(df['charges'], kde=True, ax=ax, color='teal')
        ax.set_title("Distribution of Charges")
    elif viz == "Charges by Gender":
        sns.boxplot(x='sex_label', y='charges', data=df, ax=ax, palette="pastel")
        ax.set_title("Charges by Gender")
    elif viz == "Charges by Smoking Status":
        sns.boxplot(x='smoker_label', y='charges', data=df, ax=ax, palette="Set2")
        ax.set_title("Charges: Smokers vs Non-Smokers")
    elif viz == "Charges vs BMI":
        sns.scatterplot(x='bmi', y='charges', hue='smoker_label', data=df, ax=ax)
        ax.set_title("Charges vs BMI")
    elif viz == "Charges vs Age":
        sns.scatterplot(x='age', y='charges', hue='smoker_label', data=df, ax=ax)
        ax.set_title("Charges vs Age")
    elif viz == "Region-wise Policyholders":
        sns.countplot(x='region_label', data=df, ax=ax, palette='Accent')
        ax.set_title("Policyholders by Region")
    elif viz == "Children vs Charges":
        avg_charges = df.groupby('children')['charges'].mean().reindex(range(6), fill_value=0)
        sns.barplot(x=avg_charges.index, y=avg_charges.values, ax=ax, palette="coolwarm")
        ax.set_title("Average Charges by Number of Children")
    st.pyplot(fig)

# Prediction Page
elif page == "💰 Prediction":
    st.title("💰 Predict Insurance Charges")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 100, 30)
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
            children = st.slider("Children", 0, 5, 0)
        with col2:
            sex = st.selectbox("Gender", ["Female", "Male"])
            smoker = st.selectbox("Smoker", ["No", "Yes"])
            region = st.selectbox("Region", ["Northeast", "Southeast", "Southwest", "Northwest"])
        submitted = st.form_submit_button("Predict")

    if submitted:
        sex_encoded = 1 if sex == "Male" else 0
        smoker_encoded = 1 if smoker == "Yes" else 0
        region_encoded = {"Northeast": 0, "Southeast": 1, "Southwest": 2, "Northwest": 3}[region]
        input_df = pd.DataFrame([[age, bmi, children, sex_encoded, smoker_encoded, region_encoded]],
                                columns=['age', 'bmi', 'children', 'sex', 'smoker', 'region'])
        try:
            pred = model.predict(input_df)[0]
            st.success(f"### 💰 Predicted Insurance Charge: ${pred:,.2f}")
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
