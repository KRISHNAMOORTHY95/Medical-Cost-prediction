import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

# Configure Streamlit page
st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="🏥", layout="wide")


# === Caching the Model ===
@st.cache_resource
def load_or_create_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning("Model file not found. Training a sample model...")

        np.random.seed(42)
        n = 1000
        df_sample = pd.DataFrame({
            'age': np.random.randint(18, 65, n),
            'bmi': np.clip(np.random.normal(28, 6, n), 15, 50),
            'children': np.random.randint(0, 6, n),
            'smoker': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'sex': np.random.choice([0, 1], n),
            'region': np.random.choice([0, 1, 2, 3], n)
        })
        df_sample['charges'] = (
            df_sample['age'] * 200 +
            df_sample['bmi'] * 100 +
            df_sample['children'] * 500 +
            df_sample['smoker'] * 15000 +
            df_sample['sex'] * 200 +
            df_sample['region'] * 300 +
            np.random.normal(0, 2000, n)
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(df_sample.drop('charges', axis=1), df_sample['charges'])

        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

        return model


# === Caching the Dataset ===
@st.cache_data
def load_or_create_dataset():
    try:
        df = pd.read_csv("medical_insurance.csv")
        return df
    except FileNotFoundError:
        st.warning("Dataset not found. Generating sample dataset.")
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'age': np.random.randint(18, 65, n),
            'bmi': np.clip(np.random.normal(28, 6, n), 15, 50),
            'children': np.random.randint(0, 6, n),
            'smoker': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'sex': np.random.choice([0, 1], n),
            'region': np.random.choice([0, 1, 2, 3], n)
        })
        df['charges'] = (
            df['age'] * 200 +
            df['bmi'] * 100 +
            df['children'] * 500 +
            df['smoker'] * 15000 +
            df['sex'] * 200 +
            df['region'] * 300 +
            np.random.normal(0, 2000, n)
        )
        return df


# === Load Data and Model ===
df = load_or_create_dataset()
model = load_or_create_model()

# === Sidebar Navigation ===
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Introduction", "📊 Visualizations", "💰 Cost Prediction"])

# === Page: Introduction ===
if page == "🏠 Introduction":
    st.title("🏥 Medical Insurance Cost Prediction")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## Welcome!

        This app uses machine learning to estimate medical insurance charges based on key factors.

        ### 🔍 Features:
        - Explore relationships between features like BMI, smoking, region, and insurance costs
        - Predict charges based on user inputs
        - Powered by Random Forest Regressor
        """)

    with col2:
        st.info(f"📄 Records: {len(df):,}")
        st.info(f"💲 Avg. Charge: ${df['charges'].mean():,.2f}")
        st.info(f"💲 Max: ${df['charges'].max():,.2f}")
        st.info(f"💲 Min: ${df['charges'].min():,.2f}")

# Page 2: Visualizations
elif page == "📊 Visualizations":
    st.title("📊 Exploratory Data Analysis")

    # Always work on a copy of the data to avoid persistent modifications
    data = df.copy()

    def show_distribution_of_charges():
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data['charges'], kde=True, bins=30, color='teal', ax=ax)
        ax.set_title("Distribution of Charges")
        return fig

    def show_age_distribution():
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data['age'], kde=True, bins=30, color='skyblue', ax=ax)
        ax.set_title("Age Distribution")
        return fig

    def show_bmi_distribution():
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data['bmi'], kde=True, bins=30, color='orchid', ax=ax)
        ax.set_title("BMI Distribution")
        return fig

    def show_smoker_counts():
        temp = data.copy()
        temp['smoker_status'] = temp['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.countplot(x='smoker_status', data=temp, palette='Set2', ax=ax)
        ax.set_title("Smoker vs Non-Smoker Count")
        return fig

    def show_region_counts():
        temp = data.copy()
        region_map = {0: 'Northeast', 1: 'Southeast', 2: 'Southwest', 3: 'Northwest'}
        temp['region_label'] = temp['region'].map(region_map)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='region_label', data=temp, palette='pastel', ax=ax)
        ax.set_title("Policyholders by Region")
        return fig

    def show_charge_age():
        temp = data.copy()
        temp['smoker_status'] = temp['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='age', y='charges', hue='smoker_status', data=temp, palette='Set1', ax=ax)
        ax.set_title("Charges vs Age by Smoking Status")
        return fig

    def show_charge_bmi():
        temp = data.copy()
        temp['smoker_status'] = temp['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='bmi', y='charges', hue='smoker_status', data=temp, palette='Set1', ax=ax)
        ax.set_title("Charges vs BMI by Smoking Status")
        return fig

    def show_gender_charges():
        temp = data.copy()
        temp['gender'] = temp['sex'].map({0: 'Female', 1: 'Male'})
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='gender', y='charges', data=temp, palette='pastel', ax=ax)
        ax.set_title("Medical Charges by Gender", fontsize=14, fontweight='bold')
        ax.set_xlabel("Gender", fontsize=12)
        ax.set_ylabel("Charges ($)", fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        return fig


    def show_children_charges():
        fig, ax = plt.subplots(figsize=(8, 5))
        avg_charges = data.groupby('children')['charges'].mean().reindex(range(0, 6), fill_value=0)
        sns.barplot(x=avg_charges.index, y=avg_charges.values, palette='coolwarm', ax=ax)
        ax.set_title("Average Charges by Number of Children")
        return fig

    def show_smoker_charges_box():
        temp = data.copy()
        temp['smoker_status'] = temp['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(x='smoker_status', y='charges', data=temp, palette='Set2', ax=ax)
        ax.set_title("Charges: Smoker vs Non-Smoker")
        return fig

    def show_correlation_matrix():
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = data[['age', 'bmi', 'children', 'charges']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', square=True, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        return fig

    # Visualization selector
    visualizations = {
        "📈 Distribution of Charges": show_distribution_of_charges,
        "👥 Age Distribution": show_age_distribution,
        "⚖️ BMI Distribution": show_bmi_distribution,
        "🚭 Smoker Count": show_smoker_counts,
        "🗺️ Region Count": show_region_counts,
        "📊 Charges vs Age": show_charge_age,
        "📉 Charges vs BMI": show_charge_bmi,
        "👫 Charges by Gender": show_gender_charges,
        "👶 Charges vs Number of Children": show_children_charges,
        "💰 Charges: Smoker vs Non-Smoker": show_smoker_charges_box,
        "🔗 Feature Correlation Matrix": show_correlation_matrix
    }

    # User selection
    selected_viz = st.selectbox("📊 Choose a Visualization", list(visualizations.keys()))

    # Show plot
    try:
        fig = visualizations[selected_viz]()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"⚠️ Error showing plot: {e}")

# === Page: Prediction ===
elif page == "💰 Cost Prediction":
    st.title("💰 Predict Insurance Charges")

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 100, 30)
            bmi = st.slider("BMI", 10.0, 60.0, 25.0)
            children = st.selectbox("Number of Children", list(range(0, 6)))
        with col2:
            sex = st.radio("Gender", ["Male", "Female"])
            smoker = st.radio("Smoker", ["Yes", "No"])
            region = st.selectbox("Region", ["Northeast", "Southeast", "Southwest", "Northwest"])

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        input_df = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex': [1 if sex == "Male" else 0],
            'smoker': [1 if smoker == "Yes" else 0],
            'region': [ {"Northeast": 0, "Southeast": 1, "Southwest": 2, "Northwest": 3}[region] ]
        })

        try:
            prediction = model.predict(input_df)[0]
            st.success(f"💡 Estimated Insurance Cost: ${prediction:,.2f}")
            st.subheader("📋 Summary")
            st.table(input_df.rename(columns={
                'age': 'Age', 'bmi': 'BMI', 'children': 'Children',
                'sex': 'Gender (0=F,1=M)', 'smoker': 'Smoker (0=No,1=Yes)', 'region': 'Region (0–3)'
            }))
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.dataframe(input_df)

# === Footer ===
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ using Streamlit & ML")
