import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Medical Insurance Cost Prediction", layout="wide")

# Function to load and prepare data
@st.cache_data
def load_data():
    """
    Load the medical insurance dataset.
    If the file doesn't exist, create sample data for demonstration.
    """
    try:
        # Try to load the actual dataset
        df = pd.read_csv("medical_insurance.csv")
    except FileNotFoundError:
        # Create sample data if file doesn't exist
        np.random.seed(42)
        n_samples = 1000
        
        ages = np.random.randint(18, 65, n_samples)
        bmis = np.random.normal(28, 6, n_samples)
        bmis = np.clip(bmis, 15, 50)  # Clip BMI to reasonable range
        children = np.random.poisson(1.2, n_samples)
        children = np.clip(children, 0, 5)
        
        sexes = np.random.choice(['male', 'female'], n_samples)
        smokers = np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8])
        regions = np.random.choice(['northeast', 'southeast', 'southwest', 'northwest'], n_samples)
        
        # Generate charges based on realistic factors
        base_charges = 3000 + ages * 100 + bmis * 200 + children * 500
        smoker_multiplier = np.where(np.array(smokers) == 'yes', 3, 1)
        charges = base_charges * smoker_multiplier + np.random.normal(0, 2000, n_samples)
        charges = np.maximum(charges, 1000)  # Minimum charge
        
        df = pd.DataFrame({
            'age': ages,
            'sex': sexes,
            'bmi': bmis,
            'children': children,
            'smoker': smokers,
            'region': regions,
            'charges': charges
        })
    
    return df

@st.cache_data
def prepare_data(df):
    """Prepare data for modeling"""
    df_processed = df.copy()
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    df_processed['sex'] = le_sex.fit_transform(df_processed['sex'])
    df_processed['smoker'] = le_smoker.fit_transform(df_processed['smoker'])
    df_processed['region'] = le_region.fit_transform(df_processed['region'])
    
    return df_processed, le_sex, le_smoker, le_region

@st.cache_resource
def train_model(df_processed):
    """Train the Random Forest model"""
    X = df_processed[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    y = df_processed['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Load data and prepare model
df = load_data()
df_processed, le_sex, le_smoker, le_region = prepare_data(df)
model = train_model(df_processed)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Introduction", "Visualizations", "Cost Prediction"])

# Page 1: Project Introduction
if page == "Project Introduction":
    st.title("🏥 Medical Insurance Cost Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Project Overview
        This application analyzes medical insurance data to understand the factors that influence insurance costs 
        and provides predictions for new cases using machine learning.
        
        **Key Features:**
        - **Exploratory Data Analysis**: Interactive visualizations to understand data patterns
        - **Cost Prediction**: ML-powered insurance cost estimation
        - **Insights**: Discover which factors most significantly impact insurance charges
        
        **Dataset Features:**
        - **Age**: Age of the policyholder
        - **Sex**: Gender of the policyholder
        - **BMI**: Body Mass Index
        - **Children**: Number of dependents
        - **Smoker**: Smoking status
        - **Region**: Geographic region
        - **Charges**: Insurance charges (target variable)
        """)
    
    with col2:
        st.markdown("### Dataset Summary")
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Average Age:** {df['age'].mean():.1f} years")
        st.write(f"**Average BMI:** {df['bmi'].mean():.1f}")
        st.write(f"**Average Charges:** ${df['charges'].mean():,.2f}")
        st.write(f"**Smokers:** {(df['smoker'] == 'yes').sum():,} ({(df['smoker'] == 'yes').mean()*100:.1f}%)")

# Page 2: Visualizations
elif page == "Visualizations":
    st.title("📊 Exploratory Data Analysis")
    
    # Create visualization functions
    def plot_charges_distribution():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['charges'], kde=True, bins=30, color='teal', ax=ax)
        ax.set_title('Distribution of Medical Insurance Charges', fontsize=16)
        ax.set_xlabel('Charges ($)')
        ax.set_ylabel('Frequency')
        return fig
    
    def plot_age_distribution():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['age'], kde=True, bins=20, color='skyblue', ax=ax)
        ax.set_title('Distribution of Age', fontsize=16)
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        return fig
    
    def plot_smoker_count():
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=df, x='smoker', palette='Set2', ax=ax)
        ax.set_title('Count of Smokers vs Non-Smokers', fontsize=16)
        ax.set_xlabel('Smoking Status')
        ax.set_ylabel('Number of Individuals')
        return fig
    
    def plot_bmi_distribution():
        fig, ax = plt.subplots(figsize=(10, 6))
        average_bmi = df['bmi'].mean()
        sns.histplot(df['bmi'], kde=True, bins=30, color='purple', ax=ax)
        ax.axvline(average_bmi, color='red', linestyle='--', label=f'Mean BMI: {average_bmi:.2f}')
        ax.set_title('Distribution of BMI', fontsize=16)
        ax.set_xlabel('BMI')
        ax.set_ylabel('Frequency')
        ax.legend()
        return fig
    
    def plot_region_count():
        fig, ax = plt.subplots(figsize=(10, 6))
        region_counts = df['region'].value_counts()
        sns.countplot(data=df, x='region', palette='Set3', order=region_counts.index, ax=ax)
        ax.set_title('Number of Policyholders by Region', fontsize=16)
        ax.set_xlabel('Region')
        ax.set_ylabel('Number of Policyholders')
        return fig
    
    def plot_charges_vs_age():
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=df, x='age', y='charges', hue='smoker', alpha=0.6, palette='Set1', ax=ax)
        ax.set_title('Charges vs. Age (Colored by Smoker Status)', fontsize=16)
        ax.set_xlabel('Age')
        ax.set_ylabel('Charges ($)')
        return fig
    
    def plot_charges_smoker_box():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='smoker', y='charges', palette='Set2', ax=ax)
        ax.set_title('Medical Charges: Smokers vs Non-Smokers', fontsize=16)
        ax.set_xlabel('Smoking Status')
        ax.set_ylabel('Charges ($)')
        return fig
    
    def plot_charges_vs_bmi():
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', alpha=0.6, palette='Set1', ax=ax)
        ax.set_title('Charges vs. BMI (Colored by Smoker Status)', fontsize=16)
        ax.set_xlabel('BMI')
        ax.set_ylabel('Charges ($)')
        return fig
    
    def plot_charges_by_gender():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='sex', y='charges', palette='pastel', ax=ax)
        ax.set_title('Medical Charges: Male vs Female', fontsize=16)
        ax.set_xlabel('Gender')
        ax.set_ylabel('Charges ($)')
        return fig
    
    def plot_charges_by_children():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x='children', y='charges', estimator='mean', palette='coolwarm', ax=ax)
        ax.set_title('Average Charges by Number of Children', fontsize=16)
        ax.set_xlabel('Number of Children')
        ax.set_ylabel('Average Charges ($)')
        return fig
    
    def plot_correlation_heatmap():
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_cols = ['age', 'bmi', 'children', 'charges']
        corr_matrix = df_processed[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title('Correlation Between Numeric Features', fontsize=16)
        return fig
    
    def plot_feature_importance():
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance = model.feature_importances_
        features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        importance_df = pd.DataFrame({'feature': features, 'importance': feature_importance})
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis', ax=ax)
        ax.set_title('Feature Importance in Random Forest Model', fontsize=16)
        ax.set_xlabel('Importance')
        ax.set_ylabel('Features')
        return fig
    
    # Visualization selection
    visualizations = {
        "1. Distribution of Charges": plot_charges_distribution,
        "2. Age Distribution": plot_age_distribution,
        "3. Smokers vs Non-Smokers Count": plot_smoker_count,
        "4. BMI Distribution": plot_bmi_distribution,
        "5. Policyholders by Region": plot_region_count,
        "6. Charges vs Age": plot_charges_vs_age,
        "7. Charges: Smokers vs Non-Smokers": plot_charges_smoker_box,
        "8. Charges vs BMI": plot_charges_vs_bmi,
        "9. Charges by Gender": plot_charges_by_gender,
        "10. Charges by Number of Children": plot_charges_by_children,
        "11. Correlation Heatmap": plot_correlation_heatmap,
        "12. Feature Importance": plot_feature_importance,
    }
    
    selected_viz = st.selectbox("Select a visualization:", list(visualizations.keys()))
    
    if st.button("Generate Visualization"):
        fig = visualizations[selected_viz]()
        st.pyplot(fig)
        plt.close(fig)  # Close figure to free memory

# Page 3: Cost Prediction
elif page == "Cost Prediction":
    st.title("💰 Insurance Cost Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Enter Your Details:")
        
        age = st.slider("Age", min_value=18, max_value=100, value=30)
        bmi = st.slider("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        children = st.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5])
        smoker = st.selectbox("Smoking Status", ["no", "yes"])
        region = st.selectbox("Region", ['northeast', 'southeast', 'southwest', 'northwest'])
        sex = st.selectbox("Gender", ["male", "female"])
    
    with col2:
        st.subheader("Prediction Results:")
        
        if st.button("🔮 Predict Insurance Cost", type="primary"):
            # Encode inputs
            sex_encoded = le_sex.transform([sex])[0]
            smoker_encoded = le_smoker.transform([smoker])[0]
            region_encoded = le_region.transform([region])[0]
            
            # Create input DataFrame
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex_encoded],
                'bmi': [bmi],
                'children': [children],
                'smoker': [smoker_encoded],
                'region': [region_encoded]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display results
            st.success(f"💵 **Predicted Insurance Cost: ${prediction:,.2f}**")
            
            # Show input summary
            st.write("**Input Summary:**")
            summary_data = {
                "Age": age,
                "Gender": sex.title(),
                "BMI": f"{bmi:.1f}",
                "Children": children,
                "Smoker": smoker.title(),
                "Region": region.title()
            }
            
            for key, value in summary_data.items():
                st.write(f"- **{key}:** {value}")
            
            # Show similar cases from dataset
            st.write("**Similar Cases in Dataset:**")
            similar_cases = df[
                (abs(df['age'] - age) <= 5) & 
                (df['smoker'] == smoker) & 
                (df['sex'] == sex)
            ].head(3)
            
            if not similar_cases.empty:
                st.dataframe(similar_cases[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']])
            else:
                st.write("No similar cases found in the dataset.")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This app uses Random Forest Regression to predict medical insurance costs based on personal factors.

**Model Performance:**
- Uses ensemble learning
- Handles non-linear relationships
- Provides feature importance insights
""")
