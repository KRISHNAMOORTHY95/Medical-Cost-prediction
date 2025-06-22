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
    
    # Visualization functions
    def show_distribution_of_charges():
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df['charges'], kde=True, bins=30, color='teal', alpha=0.7, ax=ax)
        ax.set_title('Distribution of Medical Insurance Charges', fontsize=16, fontweight='bold')
        ax.set_xlabel('Charges ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        return fig

    def show_age_distribution():
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df['age'], kde=True, bins=20, color='skyblue', alpha=0.7, ax=ax)
        ax.set_title('Distribution of Age', fontsize=16, fontweight='bold')
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        return fig

    def show_smoker_non_smoker():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create smoker labels
        df['smoker_label'] = df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        smoker_counts = df['smoker_label'].value_counts().reindex(['Non-Smoker', 'Smoker'], fill_value=0)
        
        # Create count plot
        sns.countplot(x='smoker_label', data=df, palette='Set2', order=['Non-Smoker', 'Smoker'], ax=ax)
        
        # Styling
        ax.set_title('Count of Smokers vs Non-Smokers', fontsize=16, fontweight='bold')
        ax.set_xlabel('Smoking Status', fontsize=12)
        ax.set_ylabel('Number of Individuals', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Add percentage labels
        total = len(df)
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            percentage = (height / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2, height/2,
                   f'{percentage:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=12)
        
        return fig

    def show_avg_bmi():
        fig, ax = plt.subplots(figsize=(12, 6))
        average_bmi = df['bmi'].mean()
        sns.histplot(df['bmi'], kde=True, bins=30, color='purple', alpha=0.7, ax=ax)
        ax.axvline(average_bmi, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean BMI: {average_bmi:.2f}')
        ax.set_title('Distribution of BMI', fontsize=16, fontweight='bold')
        ax.set_xlabel('BMI', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig

    def show_no_of_policyholders():
        fig, ax = plt.subplots(figsize=(10, 6))
        region_counts = df['region'].value_counts().sort_index()
        region_labels = ['Northeast', 'Southeast', 'Southwest', 'Northwest']
        
        # Create color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(region_counts)))
        
        # Create bar chart
        bars = ax.bar(region_labels, region_counts.values, color=colors, alpha=0.8)
        
        # Styling
        ax.set_title('Number of Policyholders by Region', fontsize=16, fontweight='bold')
        ax.set_xlabel('Region', fontsize=12)
        ax.set_ylabel('Number of Policyholders', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, region_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig

    def show_charge_age():
        fig, ax = plt.subplots(figsize=(12, 8))
        smoker_labels = df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        sns.scatterplot(x='age', y='charges', hue=smoker_labels, data=df.assign(smoker=smoker_labels), 
                       alpha=0.6, palette='Set1', s=60, ax=ax)
        ax.set_title('Charges vs. Age (Colored by Smoker Status)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Charges ($)', fontsize=12)
        ax.legend(title='Smoking Status')
        ax.grid(True, alpha=0.3)
        return fig

    def show_charges_smokervsnon():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create smoker labels
        df['smoker_label'] = df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        
        # Create boxplot with enhanced styling
        sns.boxplot(x='smoker_label', y='charges', data=df, palette='Set2', 
                   order=['Non-Smoker', 'Smoker'], ax=ax)
        
        # Styling
        ax.set_title('Medical Charges: Smokers vs Non-Smokers', fontsize=16, fontweight='bold')
        ax.set_xlabel('Smoking Status', fontsize=12)
        ax.set_ylabel('Charges ($)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format y-axis to show currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Calculate statistics
        nonsmoker_data = df[df['smoker_label'] == 'Non-Smoker']['charges']
        smoker_data = df[df['smoker_label'] == 'Smoker']['charges']
        
        nonsmoker_median = nonsmoker_data.median()
        smoker_median = smoker_data.median()
        nonsmoker_mean = nonsmoker_data.mean()
        smoker_mean = smoker_data.mean()
        
        # Calculate the difference
        median_diff = smoker_median - nonsmoker_median
        mean_diff = smoker_mean - nonsmoker_mean
        
        # Add statistical annotations
        stats_text = f'Non-Smoker - Median: ${nonsmoker_median:,.0f}, Mean: ${nonsmoker_mean:,.0f}\n'
        stats_text += f'Smoker - Median: ${smoker_median:,.0f}, Mean: ${smoker_mean:,.0f}\n'
        stats_text += f'Difference - Median: ${median_diff:,.0f}, Mean: ${mean_diff:,.0f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Add sample size information
        nonsmoker_count = len(nonsmoker_data)
        smoker_count = len(smoker_data)
        
        ax.text(0, -0.08, f'n = {nonsmoker_count}', transform=ax.transData, 
               ha='center', va='top', fontweight='bold')
        ax.text(1, -0.08, f'n = {smoker_count}', transform=ax.transData, 
               ha='center', va='top', fontweight='bold')
        
        # Add a striking visual indicator of the difference
        ax.annotate('', xy=(1, smoker_median), xytext=(0, nonsmoker_median),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(0.5, (smoker_median + nonsmoker_median) / 2, 
               f'${median_diff:,.0f}\nDifference', 
               ha='center', va='center', fontweight='bold', color='red',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return fig

    def show_bmi_charge():
        fig, ax = plt.subplots(figsize=(12, 8))
        smoker_labels = df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        sns.scatterplot(x='bmi', y='charges', hue=smoker_labels, data=df.assign(smoker=smoker_labels), 
                       alpha=0.6, palette='Set1', s=60, ax=ax)
        ax.set_title('Charges vs. BMI (Colored by Smoker Status)', fontsize=16, fontweight='bold')
        ax.set_xlabel('BMI', fontsize=12)
        ax.set_ylabel('Charges ($)', fontsize=12)
        ax.legend(title='Smoking Status')
        ax.grid(True, alpha=0.3)
        return fig

    def show_men_women_charge():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create gender labels
        df['sex_label'] = df['sex'].map({0: 'Female', 1: 'Male'})
        
        # Create boxplot
        sns.boxplot(x='sex_label', y='charges', data=df, palette='pastel', 
                   order=['Female', 'Male'], ax=ax)
        
        # Styling
        ax.set_title('Medical Charges: Male vs Female', fontsize=16, fontweight='bold')
        ax.set_xlabel('Gender', fontsize=12)
        ax.set_ylabel('Charges ($)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format y-axis to show currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add statistical annotations
        female_median = df[df['sex_label'] == 'Female']['charges'].median()
        male_median = df[df['sex_label'] == 'Male']['charges'].median()
        female_mean = df[df['sex_label'] == 'Female']['charges'].mean()
        male_mean = df[df['sex_label'] == 'Male']['charges'].mean()
        
        # Add text box with statistics
        stats_text = f'Female - Median: ${female_median:,.0f}, Mean: ${female_mean:,.0f}\n'
        stats_text += f'Male - Median: ${male_median:,.0f}, Mean: ${male_mean:,.0f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add sample size information
        female_count = len(df[df['sex_label'] == 'Female'])
        male_count = len(df[df['sex_label'] == 'Male'])
        
        ax.text(0, -0.1, f'n = {female_count}', transform=ax.transData, 
               ha='center', va='top', fontweight='bold')
        ax.text(1, -0.1, f'n = {male_count}', transform=ax.transData, 
               ha='center', va='top', fontweight='bold')
        
        return fig

    def show_correlation_children_charge():
        fig, ax = plt.subplots(figsize=(10, 6))
        children_avg = df.groupby('children')['charges'].mean().reindex(range(0, 6), fill_value=0)
        colors = plt.cm.viridis(np.linspace(0, 1, len(children_avg)))
        
        bars = ax.bar(children_avg.index, children_avg.values, color=colors, alpha=0.8)
        ax.set_title('Average Charges by Number of Children', fontsize=16, fontweight='bold')
        ax.set_xlabel('Number of Children', fontsize=12)
        ax.set_ylabel('Average Charges ($)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, avg_charge in zip(bars, children_avg.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                    f'${avg_charge:,.0f}', ha='center', va='bottom')
        return fig

    def show_numeric_features():
        fig, ax = plt.subplots(figsize=(8, 6))
        numeric_cols = ['age', 'bmi', 'children', 'charges']
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                   linewidths=0.5, square=True, cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Correlation Between Numeric Features', fontsize=16, fontweight='bold')
        return fig

    # Questions dictionary
    questions = {
        "📈 Distribution of Charges": show_distribution_of_charges,
        "👥 Age Distribution": show_age_distribution,
        "🚭 Smokers vs Non-Smokers": show_smoker_non_smoker,
        "⚖️ BMI Distribution": show_avg_bmi,
        "🗺️ Policyholders by Region": show_no_of_policyholders,
        "📊 Charges vs Age": show_charge_age,
        "💰 Charges: Smokers vs Non-Smokers": show_charges_smokervsnon,
        "📉 Charges vs BMI": show_bmi_charge,
        "👫 Charges by Gender": show_men_women_charge,
        "👶 Charges vs Number of Children": show_correlation_children_charge,
        "🔗 Feature Correlations": show_numeric_features,
    }

    # Create selectbox for visualizations
    selected_question = st.selectbox("🔍 Select a visualization:", list(questions.keys()))
    
    # Create two columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Execute selected visualization and display
        try:
            fig = questions[selected_question]()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
    
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
