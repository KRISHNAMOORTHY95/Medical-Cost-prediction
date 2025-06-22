import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Configure page
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide"
)

# Function to create and train model if not exists
@st.cache_data
def load_or_create_model():
    try:
        # Try to load existing model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning("Model file not found. Creating a new model with sample data.")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        ages = np.random.randint(18, 65, n_samples)
        bmis = np.random.normal(28, 6, n_samples)
        bmis = np.clip(bmis, 15, 50)
        children = np.random.randint(0, 6, n_samples)
        smokers = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        sexes = np.random.choice([0, 1], n_samples)
        regions = np.random.choice([0, 1, 2, 3], n_samples)
        
        # Create charges with realistic relationships
        charges = (
            ages * 200 + 
            bmis * 100 + 
            children * 500 + 
            smokers * 15000 + 
            sexes * 200 + 
            regions * 300 + 
            np.random.normal(0, 2000, n_samples)
        )
        charges = np.clip(charges, 1000, 50000)
        
        sample_df = pd.DataFrame({
            'age': ages,
            'bmi': bmis,
            'children': children,
            'sex': sexes,
            'smoker': smokers,
            'region': regions,
            'charges': charges
        })
        
        # Train model
        X = sample_df[['age', 'bmi', 'children', 'sex', 'smoker', 'region']]
        y = sample_df['charges']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        return model

# Function to load or create dataset
@st.cache_data
def load_or_create_dataset():
    try:
        # Try to load existing dataset
        df = pd.read_csv("medical_insurance.csv")
        return df
    except FileNotFoundError:
        st.warning("Dataset file not found. Creating sample data for demonstration.")
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        ages = np.random.randint(18, 65, n_samples)
        bmis = np.random.normal(28, 6, n_samples)
        bmis = np.clip(bmis, 15, 50)
        children = np.random.randint(0, 6, n_samples)
        smokers = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        sexes = np.random.choice([0, 1], n_samples)
        regions = np.random.choice([0, 1, 2, 3], n_samples)
        
        # Create charges with realistic relationships
        charges = (
            ages * 200 + 
            bmis * 100 + 
            children * 500 + 
            smokers * 15000 + 
            sexes * 200 + 
            regions * 300 + 
            np.random.normal(0, 2000, n_samples)
        )
        charges = np.clip(charges, 1000, 50000)
        
        df = pd.DataFrame({
            'age': ages,
            'bmi': bmis,
            'children': children,
            'sex': sexes,
            'smoker': smokers,
            'region': regions,
            'charges': charges
        })
        
        return df

# Load model and data
model = load_or_create_model()
df = load_or_create_dataset()

# Navigation
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Project Introduction", "📊 Visualizations", "💰 Cost Prediction"])

# Page 1: Project Introduction
if page == "🏠 Project Introduction":
    st.title("🏥 Medical Insurance Cost Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Medical Insurance Cost Prediction App!
        
        This application uses machine learning to analyze and predict medical insurance costs based on various factors.
        
        ### 🎯 Project Goals:
        - **Explore** patterns in insurance charges based on demographic and health factors
        - **Analyze** the relationship between age, BMI, smoking habits, region, and costs
        - **Predict** insurance charges using a Random Forest regression model
        
        ### 📊 Key Features:
        - **Interactive Visualizations**: Explore data through various charts and graphs
        - **Cost Prediction**: Get instant predictions for insurance costs
        - **Comprehensive Analysis**: Understand which factors most influence insurance costs
        
        ### 🔍 Factors Analyzed:
        - Age
        - Body Mass Index (BMI)
        - Number of children
        - Smoking status
        - Gender
        - Geographic region
        """)
    
    with col2:
        st.markdown("### 📈 Dataset Overview")
        st.info(f"**Total Records**: {len(df):,}")
        st.info(f"**Average Charge**: ${df['charges'].mean():,.2f}")
        st.info(f"**Max Charge**: ${df['charges'].max():,.2f}")
        st.info(f"**Min Charge**: ${df['charges'].min():,.2f}")

# Page 2: Visualizations
elif page == "📊 Visualizations":
    st.title("📊 Exploratory Data Analysis")
    
    # Helper function to create and display plots
    def create_plot(plot_func, title):
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_func()
        st.pyplot(fig)
        plt.close()

    # Visualization functions
    def distribution_of_charges():
        plt.figure(figsize=(12, 6))
        sns.histplot(df['charges'], kde=True, bins=30, color='teal', alpha=0.7)
        plt.title('Distribution of Medical Insurance Charges', fontsize=16, fontweight='bold')
        plt.xlabel('Charges ($)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)

    def age_distribution():
        plt.figure(figsize=(12, 6))
        sns.histplot(df['age'], kde=True, bins=20, color='skyblue', alpha=0.7)
        plt.title('Distribution of Age', fontsize=16, fontweight='bold')
        plt.xlabel('Age', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)

    def smoker_non_smoker():
        plt.figure(figsize=(8, 6))
        smoker_counts = df['smoker'].value_counts()
        labels = ['Non-Smoker', 'Smoker']
        sns.countplot(x='smoker', data=df, palette='Set2')
        plt.title('Count of Smokers vs Non-Smokers', fontsize=16, fontweight='bold')
        plt.xlabel('Smoking Status', fontsize=12)
        plt.ylabel('Number of Individuals', fontsize=12)
        plt.xticks([0, 1], labels)

    def avg_bmi():
        average_bmi = df['bmi'].mean()
        plt.figure(figsize=(12, 6))
        sns.histplot(df['bmi'], kde=True, bins=30, color='purple', alpha=0.7)
        plt.axvline(average_bmi, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean BMI: {average_bmi:.2f}')
        plt.title('Distribution of BMI', fontsize=16, fontweight='bold')
        plt.xlabel('BMI', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

    def no_of_policyholders():
        region_labels = ['Northeast', 'Southeast', 'Southwest', 'Northwest']
        region_counts = df['region'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072']  # Set3 colors
        bars = plt.bar(region_labels, region_counts.values, color=colors, alpha=0.8)
        plt.title('Number of Policyholders by Region', fontsize=16, fontweight='bold')
        plt.xlabel('Region', fontsize=12)
        plt.ylabel('Number of Policyholders', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, region_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(count), ha='center', va='bottom')

    def charge_age():
        plt.figure(figsize=(12, 8))
        smoker_labels = df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        sns.scatterplot(x='age', y='charges', hue=smoker_labels, data=df.assign(smoker=smoker_labels), 
                       alpha=0.6, palette='Set1', s=60)
        plt.title('Charges vs. Age (Colored by Smoker Status)', fontsize=16, fontweight='bold')
        plt.xlabel('Age', fontsize=12)
        plt.ylabel('Charges ($)', fontsize=12)
        plt.legend(title='Smoking Status')
        plt.grid(True, alpha=0.3)

    def charges_smokervsnon():
        plt.figure(figsize=(10, 6))
        smoker_labels = df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        sns.boxplot(x=smoker_labels, y='charges', data=df.assign(smoker=smoker_labels), palette='Set2')
        plt.title('Medical Charges: Smokers vs Non-Smokers', fontsize=16, fontweight='bold')
        plt.xlabel('Smoking Status', fontsize=12)
        plt.ylabel('Charges ($)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')

    def bmi_charge():
        plt.figure(figsize=(12, 8))
        smoker_labels = df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        sns.scatterplot(x='bmi', y='charges', hue=smoker_labels, data=df.assign(smoker=smoker_labels), 
                       alpha=0.6, palette='Set1', s=60)
        plt.title('Charges vs. BMI (Colored by Smoker Status)', fontsize=16, fontweight='bold')
        plt.xlabel('BMI', fontsize=12)
        plt.ylabel('Charges ($)', fontsize=12)
        plt.legend(title='Smoking Status')
        plt.grid(True, alpha=0.3)

    def men_women_charge():
        plt.figure(figsize=(10, 6))
        sex_labels = df['sex'].map({0: 'Female', 1: 'Male'})
        sns.boxplot(x=sex_labels, y='charges', data=df.assign(sex=sex_labels), palette='pastel')
        plt.title('Medical Charges: Male vs Female', fontsize=16, fontweight='bold')
        plt.xlabel('Gender', fontsize=12)
        plt.ylabel('Charges ($)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')

    def correlation_children_charge():
        plt.figure(figsize=(10, 6))
        children_avg = df.groupby('children')['charges'].mean()
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(children_avg)))
        bars = plt.bar(children_avg.index, children_avg.values, color=colors, alpha=0.8)
        plt.title('Average Charges by Number of Children', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Children', fontsize=12)
        plt.ylabel('Average Charges ($)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, avg_charge in zip(bars, children_avg.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                    f'${avg_charge:,.0f}', ha='center', va='bottom')

    def numeric_features():
        numeric_cols = ['age', 'bmi', 'children', 'charges']
        corr_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                   linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
        plt.title('Correlation Between Numeric Features', fontsize=16, fontweight='bold')

    # Questions dictionary
    questions = {
        "📈 Distribution of Charges": distribution_of_charges,
        "👥 Age Distribution": age_distribution,
        "🚭 Smokers vs Non-Smokers": smoker_non_smoker,
        "⚖️ BMI Distribution": avg_bmi,
        "🗺️ Policyholders by Region": no_of_policyholders,
        "📊 Charges vs Age": charge_age,
        "💰 Charges: Smokers vs Non-Smokers": charges_smokervsnon,
        "📉 Charges vs BMI": bmi_charge,
        "👫 Charges by Gender": men_women_charge,
        "👶 Charges vs Number of Children": correlation_children_charge,
        "🔗 Feature Correlations": numeric_features,
    }

    # Create selectbox for visualizations
    selected_question = st.selectbox("🔍 Select a visualization:", list(questions.keys()))
    
    # Create two columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Execute selected visualization
        questions[selected_question]()
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("### 💡 Insights")
        if "Distribution of Charges" in selected_question:
            st.info("Most insurance charges are concentrated in the lower range, with some high-cost outliers.")
        elif "Smokers" in selected_question:
            st.warning("Smokers typically have significantly higher insurance costs.")
        elif "BMI" in selected_question:
            st.info("Higher BMI combined with smoking leads to the highest charges.")
        elif "Age" in selected_question:
            st.info("Age has a positive correlation with charges, especially for smokers.")
        elif "Gender" in selected_question:
            st.info("Gender shows minimal impact on insurance charges.")
        elif "Children" in selected_question:
            st.info("Number of children has a moderate impact on insurance costs.")

# Page 3: Prediction
elif page == "💰 Cost Prediction":
    st.title("💰 Predict Insurance Charges")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Patient Details:")
        
        # Create input form
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                age = st.number_input("👤 Age", min_value=18, max_value=100, value=30, help="Patient's age in years")
                bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, 
                                    help="Body Mass Index")
                children = st.number_input("👶 Number of Children", min_value=0, max_value=10, value=0, 
                                         help="Number of dependent children")
            
            with col_b:
                smoker = st.selectbox("🚭 Smoker", ["No", "Yes"], help="Does the patient smoke?")
                region = st.selectbox("🗺️ Region", ['Northeast', 'Southeast', 'Southwest', 'Northwest'], 
                                    help="Geographic region")
                sex = st.selectbox("👫 Gender", ["Female", "Male"], help="Patient's gender")
            
            submitted = st.form_submit_button("🔮 Predict Insurance Cost", use_container_width=True)
        
        if submitted:
            # Encode inputs
            region_map = {'Northeast': 0, 'Southeast': 1, 'Southwest': 2, 'Northwest': 3}
            region_encoded = region_map[region]
            sex_encoded = 1 if sex == 'Male' else 0
            smoker_encoded = 1 if smoker == 'Yes' else 0
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'bmi': [bmi],
                'children': [children],
                'sex': [sex_encoded],
                'smoker': [smoker_encoded],
                'region': [region_encoded]
            })
            
            # Make prediction
            try:
                prediction = model.predict(input_data)[0]
                
                # Display prediction
                st.success(f"### 💰 Predicted Insurance Cost: ${prediction:,.2f}")
                
                # Show input summary
                st.subheader("📋 Input Summary:")
                summary_df = pd.DataFrame({
                    'Feature': ['Age', 'BMI', 'Children', 'Gender', 'Smoker', 'Region'],
                    'Value': [age, f"{bmi:.1f}", children, sex, smoker, region]
                })
                st.table(summary_df)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with col2:
        st.markdown("### 💡 Prediction Tips")
        st.info("**Age**: Older individuals typically have higher insurance costs")
        st.info("**BMI**: Higher BMI may increase costs")
        st.warning("**Smoking**: This is the biggest factor affecting insurance costs")
        st.info("**Children**: More dependents usually increase costs")
        st.info("**Region**: Different regions have varying cost structures")
        
        # Show model info
        st.markdown("### 🤖 Model Information")
        st.success("**Model Type**: Random Forest Regressor")
        st.success("**Features Used**: 6 key factors")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About This App")
st.sidebar.info("This app demonstrates machine learning for insurance cost prediction using demographic and health factors.")
