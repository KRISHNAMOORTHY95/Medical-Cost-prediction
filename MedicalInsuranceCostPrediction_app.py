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
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for better Streamlit compatibility
plt.style.use('default')
sns.set_palette("husl")

# Configure page
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide"
)

# Function to create sample data
def create_sample_data():
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
        
        # Create sample data
        sample_df = create_sample_data()

        # Train model
        X = sample_df[['age', 'bmi', 'children', 'sex', 'smoker', 'region']]
        y = sample_df['charges']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Try to save model (optional, may fail in some environments)
        try:
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)
        except:
            pass  # Ignore save errors

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
        return create_sample_data()

# Helper function to create matplotlib charts
def create_matplotlib_chart(chart_function):
    """Helper function to create and display matplotlib charts"""
    fig, ax = plt.subplots(figsize=(12, 6))
    chart_function(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

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

    # Streamlit native charts (most reliable)
    def show_distribution_of_charges():
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df['charges'], kde=True, bins=30, color='teal', alpha=0.7, ax=ax)
        ax.set_title('Distribution of Medical Insurance Charges', fontsize=16, fontweight='bold')
        ax.set_xlabel('Charges ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    def show_age_distribution_native():
        """Fixed function for age distribution using native plotting"""
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df['age'], kde=True, bins=20, color='skyblue', alpha=0.7, ax=ax)
        ax.set_title('Distribution of Age', fontsize=16, fontweight='bold')
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def smoker_count_chart(ax):
        smoker_counts = df['smoker'].value_counts()
        labels = ['Non-Smoker', 'Smoker']
        colors = ['lightblue', 'salmon']
        
        bars = ax.bar(labels, [smoker_counts.get(0, 0), smoker_counts.get(1, 0)], 
                     color=colors, alpha=0.8)
        ax.set_title('Count of Smokers vs Non-Smokers', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{int(height)}', ha='center', va='bottom')

    def bmi_distribution_chart(ax):
        ax.hist(df['bmi'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(df['bmi'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df["bmi"].mean():.1f}')
        ax.set_title('BMI Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('BMI')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def region_chart(ax):
        region_map = {0: 'Northeast', 1: 'Southeast', 2: 'Southwest', 3: 'Northwest'}
        region_counts = df['region'].map(region_map).value_counts()
        
        if len(region_counts) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
            
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax.bar(range(len(region_counts)), region_counts.values, 
                     color=colors[:len(region_counts)], alpha=0.8)
        ax.set_title('Policyholders by Region', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(region_counts)))
        ax.set_xticklabels(region_counts.index, rotation=45)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, region_counts.values)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(region_counts.values) * 0.01,
                   f'{int(value)}', ha='center', va='bottom')

    def charges_vs_age_chart(ax):
        # Separate smokers and non-smokers
        non_smokers = df[df['smoker'] == 0]
        smokers = df[df['smoker'] == 1]
        
        if len(non_smokers) > 0:
            ax.scatter(non_smokers['age'], non_smokers['charges'], 
                      alpha=0.6, label='Non-Smoker', color='blue', s=30)
        
        if len(smokers) > 0:
            ax.scatter(smokers['age'], smokers['charges'], 
                      alpha=0.6, label='Smoker', color='red', s=30)
        
        ax.set_title('Charges vs Age by Smoking Status', fontsize=14, fontweight='bold')
        ax.set_xlabel('Age')
        ax.set_ylabel('Charges ($)')
        
        if len(non_smokers) > 0 or len(smokers) > 0:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            
        ax.grid(True, alpha=0.3)

    def smoker_charges_boxplot(ax):
        non_smoker_charges = df[df['smoker'] == 0]['charges'].dropna()
        smoker_charges = df[df['smoker'] == 1]['charges'].dropna()
        
        if len(non_smoker_charges) == 0 and len(smoker_charges) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
            
        # Prepare data for boxplot
        smoker_data = []
        labels = []
        
        if len(non_smoker_charges) > 0:
            smoker_data.append(non_smoker_charges)
            labels.append('Non-Smoker')
            
        if len(smoker_charges) > 0:
            smoker_data.append(smoker_charges)
            labels.append('Smoker')
        
        if smoker_data:
            box = ax.boxplot(smoker_data, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'salmon']
            for i, patch in enumerate(box['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
        
        ax.set_title('Charges: Smokers vs Non-Smokers', fontsize=14, fontweight='bold')
        ax.set_ylabel('Charges ($)')
        ax.grid(True, alpha=0.3)

    def charges_vs_bmi_chart(ax):
        non_smokers = df[df['smoker'] == 0]
        smokers = df[df['smoker'] == 1]
        
        if len(non_smokers) > 0:
            ax.scatter(non_smokers['bmi'], non_smokers['charges'], 
                      alpha=0.6, label='Non-Smoker', color='blue', s=30)
        
        if len(smokers) > 0:
            ax.scatter(smokers['bmi'], smokers['charges'], 
                      alpha=0.6, label='Smoker', color='red', s=30)
        
        ax.set_title('Charges vs BMI by Smoking Status', fontsize=14, fontweight='bold')
        ax.set_xlabel('BMI')
        ax.set_ylabel('Charges ($)')
        
        if len(non_smokers) > 0 or len(smokers) > 0:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            
        ax.grid(True, alpha=0.3)

    def gender_charges_chart(ax):
        female_charges = df[df['sex'] == 0]['charges'].dropna()
        male_charges = df[df['sex'] == 1]['charges'].dropna()
        
        if len(female_charges) == 0 and len(male_charges) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
            
        # Prepare data for boxplot
        gender_data = []
        labels = []
        
        if len(female_charges) > 0:
            gender_data.append(female_charges)
            labels.append('Female')
            
        if len(male_charges) > 0:
            gender_data.append(male_charges)
            labels.append('Male')
        
        if gender_data:
            box = ax.boxplot(gender_data, labels=labels, patch_artist=True)
            
            colors = ['pink', 'lightblue']
            for i, patch in enumerate(box['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
        
        ax.set_title('Charges by Gender', fontsize=14, fontweight='bold')
        ax.set_ylabel('Charges ($)')
        ax.grid(True, alpha=0.3)

    def children_charges_chart(ax):
        children_avg = df.groupby('children')['charges'].mean()
        
        bars = ax.bar(children_avg.index, children_avg.values, 
                     color='green', alpha=0.7)
        ax.set_title('Average Charges by Number of Children', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Children')
        ax.set_ylabel('Average Charges ($)')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, children_avg.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                   f'${value:,.0f}', ha='center', va='bottom')

    def correlation_heatmap(ax):
        numeric_cols = ['age', 'bmi', 'children', 'charges']
        corr_matrix = df[numeric_cols].corr()
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        
        # Add labels
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols)
        ax.set_yticklabels(numeric_cols)
        
        # Add correlation values
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha='center', va='center', color='black')
        
        ax.set_title('Feature Correlations', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Visualization options
    viz_options = {
        "📈 Distribution of Charges (Native)": show_distribution_of_charges,
        "👥 Age Distribution (Native)": show_age_distribution_native,
        "🚭 Smokers vs Non-Smokers Count": lambda: create_matplotlib_chart(smoker_count_chart),
        "⚖️ BMI Distribution": lambda: create_matplotlib_chart(bmi_distribution_chart),
        "🗺️ Policyholders by Region": lambda: create_matplotlib_chart(region_chart),
        "📊 Charges vs Age": lambda: create_matplotlib_chart(charges_vs_age_chart),
        "💰 Smoker Charges Comparison": lambda: create_matplotlib_chart(smoker_charges_boxplot),
        "📉 Charges vs BMI": lambda: create_matplotlib_chart(charges_vs_bmi_chart),
        "👫 Gender Charges Comparison": lambda: create_matplotlib_chart(gender_charges_chart),
        "👶 Children vs Charges": lambda: create_matplotlib_chart(children_charges_chart),
        "🔗 Feature Correlations": lambda: create_matplotlib_chart(correlation_heatmap),
    }

    # Visualization selector
    selected_viz = st.selectbox("🔍 Select a visualization:", list(viz_options.keys()))

    # Create columns for layout
    col1, col2 = st.columns([3, 1])

    with col1:
        # Execute selected visualization
        viz_options[selected_viz]()

    with col2:
        st.markdown("### 💡 Insights")
        if "Distribution" in selected_viz:
            st.info("Most charges are in the lower range with some high outliers.")
        elif "Smoker" in selected_viz:
            st.warning("Smoking significantly increases insurance costs.")
        elif "BMI" in selected_viz:
            st.info("Higher BMI combined with smoking leads to highest charges.")
        elif "Age" in selected_viz:
            st.info("Age correlates positively with charges, especially for smokers.")
        elif "Gender" in selected_viz:
            st.info("Gender shows minimal impact on charges.")
        elif "Children" in selected_viz:
            st.info("More children generally increase insurance costs.")
        elif "Region" in selected_viz:
            st.info("Regional differences in policyholder distribution.")
        elif "Correlation" in selected_viz:
            st.info("Shows relationships between numeric features.")

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
                age = st.number_input("👤 Age", min_value=18, max_value=100, value=30)
                bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
                children = st.number_input("👶 Number of Children", min_value=0, max_value=10, value=0)

            with col_b:
                smoker = st.selectbox("🚭 Smoker", ["No", "Yes"])
                region = st.selectbox("🗺️ Region", ['Northeast', 'Southeast', 'Southwest', 'Northwest'])
                sex = st.selectbox("👫 Gender", ["Female", "Male"])

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
        st.info("**Age**: Older individuals typically have higher costs")
        st.info("**BMI**: Higher BMI may increase costs")
        st.warning("**Smoking**: Biggest factor affecting costs")
        st.info("**Children**: More dependents usually increase costs")
        st.info("**Region**: Different regions have varying costs")

        st.markdown("### 🤖 Model Information")
        st.success("**Model**: Random Forest Regressor")
        st.success("**Features**: 6 key factors")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About This App")
st.sidebar.info("ML-powered insurance cost prediction using demographic and health factors.")
