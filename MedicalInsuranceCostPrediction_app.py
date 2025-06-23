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

# === Page: Visualizations ===
elif page == "📊 Visualizations":
    st.title("📊 Exploratory Visualizations")

    # --- Visualization functions ---
    def plot_dist(column, color, label=None, xlabel=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        temp_df = df.copy()
        sns.histplot(temp_df[column], kde=True, color=color, ax=ax)
        if label:
            ax.axvline(temp_df[column].mean(), color='red', linestyle='--', label=label)
            ax.legend()
        ax.set_title(f'{column} Distribution')
        ax.set_xlabel(xlabel or column)
        return fig

    def plot_boxplot(x, y, label_map, palette):
        fig, ax = plt.subplots(figsize=(8, 5))
        temp_df = df.copy()
        temp_df['label'] = temp_df[x].map(label_map)
        sns.boxplot(x='label', y=y, data=temp_df, palette=palette, ax=ax)
        ax.set_title(f'{y} by {x}')
        return fig

    def plot_scatter(x, y, hue, label_map):
        fig, ax = plt.subplots(figsize=(10, 6))
        temp_df = df.copy()
        temp_df[hue + '_label'] = temp_df[hue].map(label_map)
        sns.scatterplot(data=temp_df, x=x, y=y, hue=hue + '_label', palette='Set2', ax=ax)
        ax.set_title(f'{y} vs {x} by {hue}')
        return fig

    def plot_bar_count(column, label_map):
        fig, ax = plt.subplots(figsize=(8, 5))
        temp_df = df.copy()
        temp_df['label'] = temp_df[column].map(label_map)
        sns.countplot(x='label', data=temp_df, palette='Set2', ax=ax)
        ax.set_title(f'Count by {column}')
        return fig

    def plot_correlation():
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix")
        return fig

    # --- Dropdown Selection ---
    visual_options = {
        "📈 Distribution of Charges": lambda: plot_dist('charges', 'teal', "Mean", "Charges"),
        "🚻 BMI Distribution": lambda: plot_dist('bmi', 'purple', "Mean", "BMI"),
        "👵 Age Distribution": lambda: plot_dist('age', 'skyblue', "Mean", "Age"),
        "🚭 Smoker Count": lambda: plot_bar_count('smoker', {0: 'Non-Smoker', 1: 'Smoker'}),
        "🧍 Gender Charges": lambda: plot_boxplot('sex', 'charges', {0: 'Female', 1: 'Male'}, 'pastel'),
        "💨 Smoker Charges": lambda: plot_boxplot('smoker', 'charges', {0: 'Non-Smoker', 1: 'Smoker'}, 'Set2'),
        "📍 Region Policyholders": lambda: plot_bar_count('region', {0: 'NE', 1: 'SE', 2: 'SW', 3: 'NW'}),
        "💰 Charges vs BMI": lambda: plot_scatter('bmi', 'charges', 'smoker', {0: 'Non-Smoker', 1: 'Smoker'}),
        "💰 Charges vs Age": lambda: plot_scatter('age', 'charges', 'smoker', {0: 'Non-Smoker', 1: 'Smoker'}),
        "🔗 Feature Correlation": plot_correlation
    }

    choice = st.selectbox("📊 Select a chart to display:", list(visual_options.keys()))
    fig = visual_options[choice]()
    st.pyplot(fig)

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
