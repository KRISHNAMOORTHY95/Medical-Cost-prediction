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
    """
    Loads a pre-trained model from 'model.pkl'. If the file is not found,
    it creates a new Random Forest Regressor model with sample data, trains it,
    and then saves it as 'model.pkl'. This ensures the app can run even
    without a pre-existing model file.
    """
    try:
        # Try to load existing model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.warning("Model file not found. Creating a new model with sample data.")
        # Create sample data for demonstration if model is not found
        np.random.seed(42) # for reproducibility
        n_samples = 1000

        ages = np.random.randint(18, 65, n_samples)
        bmis = np.random.normal(28, 6, n_samples)
        bmis = np.clip(bmis, 15, 50) # Clip BMI to a realistic range
        children = np.random.randint(0, 6, n_samples)
        smokers = np.random.choice([0, 1], n_samples, p=[0.8, 0.2]) # 20% smokers
        sexes = np.random.choice([0, 1], n_samples) # 0 for female, 1 for male
        regions = np.random.choice([0, 1, 2, 3], n_samples) # 0,1,2,3 for regions

        # Create charges with realistic relationships
        # Charges are influenced by age, bmi, children, and significantly by smoking
        charges = (
            ages * 200 +
            bmis * 100 +
            children * 500 +
            smokers * 15000 + # Large impact for smokers
            sexes * 200 + # Slight impact for sex
            regions * 300 + # Slight impact for region
            np.random.normal(0, 2000, n_samples) # Add some random noise
        )
        charges = np.clip(charges, 1000, 50000) # Clip charges to a reasonable range

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
        st.success("New model created and saved successfully!")
        return model

# Function to load or create dataset
@st.cache_data
def load_or_create_dataset():
    """
    Loads the dataset from 'medical_insurance.csv'. If the file is not found,
    it creates a new sample DataFrame for demonstration purposes.
    """
    try:
        # Try to load existing dataset
        df = pd.read_csv("medical_insurance.csv")
        st.success("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        st.warning("Dataset file not found. Creating sample data for demonstration.")
        # Create sample data, similar to the model creation for consistency
        np.random.seed(42)
        n_samples = 1000

        ages = np.random.randint(18, 65, n_samples)
        bmis = np.random.normal(28, 6, n_samples)
        bmis = np.clip(bmis, 15, 50)
        children = np.random.randint(0, 6, n_samples)
        smokers = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        sexes = np.random.choice([0, 1], n_samples)
        regions = np.random.choice([0, 1, 2, 3], n_samples)

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
        st.success("Sample dataset created successfully!")
        return df

# Load model and data using the cached functions
model = load_or_create_model()
df = load_or_create_dataset()

# Navigation sidebar
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
        """Displays the distribution of medical insurance charges."""
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df['charges'], kde=True, bins=30, color='teal', alpha=0.7, ax=ax)
        ax.set_title('Distribution of Medical Insurance Charges', fontsize=16, fontweight='bold')
        ax.set_xlabel('Charges ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        return fig

    def show_age_distribution():
        """Displays the distribution of age in the dataset."""
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df['age'], kde=True, bins=20, color='skyblue', alpha=0.7, ax=ax)
        ax.set_title('Distribution of Age', fontsize=16, fontweight='bold')
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        return fig

    def show_smoker_non_smoker_count():
        """Displays a count of smokers vs non-smokers."""
        fig, ax = plt.subplots(figsize=(8, 6))
        # Create a temporary copy to avoid modifying the cached dataframe directly
        temp_df = df.copy()
        temp_df['smoker_label'] = temp_df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        smoker_counts = temp_df['smoker_label'].value_counts().reindex(['Non-Smoker', 'Smoker'], fill_value=0)
        sns.countplot(x='smoker_label', data=temp_df, palette='Set2', order=['Non-Smoker', 'Smoker'], ax=ax)
        ax.set_title('Count of Smokers vs Non-Smokers', fontsize=16, fontweight='bold')
        ax.set_xlabel('Smoking Status', fontsize=12)
        ax.set_ylabel('Number of Individuals', fontsize=12)
        return fig


    def show_avg_bmi():
        """Displays the distribution of BMI and highlights the mean BMI."""
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
        """Displays the number of policyholders by geographic region."""
        fig, ax = plt.subplots(figsize=(10, 6))

        region_map = {0: 'Northeast', 1: 'Southeast', 2: 'Southwest', 3: 'Northwest'}
        # Create a temporary copy to avoid modifying the cached dataframe directly
        temp_df = df.copy()
        temp_df['region_name'] = temp_df['region'].map(region_map)

        # Ensure all region names are included in the reindex, even if counts are zero
        region_counts = temp_df['region_name'].value_counts().reindex(list(region_map.values()), fill_value=0)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        bars = ax.bar(region_counts.index, region_counts.values, color=colors, alpha=0.8)
        ax.set_title('Number of Policyholders by Region', fontsize=16, fontweight='bold')
        ax.set_xlabel('Region', fontsize=12)
        ax.set_ylabel('Number of Policyholders', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, region_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                            str(count), ha='center', va='bottom')
        return fig


    def show_charge_age():
        """Displays medical charges versus age, colored by smoking status."""
        fig, ax = plt.subplots(figsize=(12, 8))
        smoker_labels = df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        # Using .assign() is a safe way to create a temporary column for plotting
        sns.scatterplot(x='age', y='charges', hue=smoker_labels, data=df.assign(smoker=smoker_labels),
                                alpha=0.6, palette='Set1', s=60, ax=ax)
        ax.set_title('Charges vs. Age (Colored by Smoker Status)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Charges ($)', fontsize=12)
        ax.legend(title='Smoking Status')
        ax.grid(True, alpha=0.3)
        return fig

    def show_charges_smokervsnon():
        """Displays a box plot comparing medical charges for smokers and non-smokers."""
        fig, ax = plt.subplots(figsize=(8, 6))
        # Create a temporary copy to avoid modifying the cached dataframe directly
        temp_df = df.copy()
        temp_df['smoker_label'] = temp_df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        # Explicitly pass an empty dictionary for boxprops to prevent potential errors
        sns.boxplot(x='smoker_label', y='charges', data=temp_df, palette='Set2',
                                order=['Non-Smoker', 'Smoker'], ax=ax, boxprops={})
        ax.set_title('Medical Charges: Smokers vs Non-Smokers', fontsize=16, fontweight='bold')
        ax.set_xlabel('Smoking Status', fontsize=12)
        ax.set_ylabel('Charges ($)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        return fig


    def show_bmi_charge():
        """Displays medical charges versus BMI, colored by smoking status."""
        fig, ax = plt.subplots(figsize=(12, 8))
        smoker_labels = df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        # Using .assign() is a safe way to create a temporary column for plotting
        sns.scatterplot(x='bmi', y='charges', hue=smoker_labels, data=df.assign(smoker=smoker_labels),
                                alpha=0.6, palette='Set1', s=60, ax=ax)
        ax.set_title('Charges vs. BMI (Colored by Smoker Status)', fontsize=16, fontweight='bold')
        ax.set_xlabel('BMI', fontsize=12)
        ax.set_ylabel('Charges ($)', fontsize=12)
        ax.legend(title='Smoking Status')
        ax.grid(True, alpha=0.3)
        return fig

    def show_men_women_charge():
        """Displays a box plot comparing medical charges for males and females."""
        fig, ax = plt.subplots(figsize=(10, 6))
        # Create a temporary copy to avoid modifying the cached dataframe directly
        temp_df = df.copy()
        temp_df['sex_label'] = temp_df['sex'].map({0: 'Female', 1: 'Male'})
        # Explicitly pass an empty dictionary for boxprops to prevent potential errors
        sns.boxplot(x='sex_label', y='charges', data=temp_df, palette='pastel',
                                order=['Female', 'Male'], ax=ax, boxprops={})

        ax.set_title('Medical Charges: Male vs Female', fontsize=16, fontweight='bold')
        ax.set_xlabel('Gender', fontsize=12)
        ax.set_ylabel('Charges ($)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        return fig

    def show_correlation_children_charge():
        """Displays average charges by the number of children."""
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
        """Displays a heatmap of correlations between numeric features."""
        fig, ax = plt.subplots(figsize=(8, 6))
        numeric_cols = ['age', 'bmi', 'children', 'charges']
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                                linewidths=0.5, square=True, cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Correlation Between Numeric Features', fontsize=16, fontweight='bold')
        return fig

    # Dictionary mapping visualization names to their functions
    questions = {
        "📈 Distribution of Charges": show_distribution_of_charges,
        "👥 Age Distribution": show_age_distribution,
        "🚭 Smokers vs Non-Smokers (Count)": show_smoker_non_smoker_count,
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

    # Create two columns for visualization and insights
    col1, col2 = st.columns([3, 1])

    with col1:
        # Execute selected visualization and display
        try:
            fig = questions[selected_question]()
            st.pyplot(fig)
            plt.close(fig) # Close the figure to prevent display issues in Streamlit
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")

    with col2:
        st.markdown("### 💡 Insights")
        # Provide context-specific insights for each visualization
        if "Distribution of Charges" in selected_question:
            st.info("Most insurance charges are concentrated in the lower range, with some high-cost outliers. This often indicates a skewed distribution, which is common in healthcare costs.")
        elif "Smokers" in selected_question:
            st.warning("Smokers typically have significantly higher insurance costs. This is a crucial factor driving up expenses, highlighting the health and financial impact of smoking.")
        elif "BMI" in selected_question:
            st.info("Higher BMI combined with smoking leads to the highest charges. While BMI alone has an impact, its combination with smoking status exacerbates costs significantly.")
        elif "Age" in selected_question:
            st.info("Age has a positive correlation with charges, especially for smokers. As individuals get older, their healthcare needs generally increase, leading to higher insurance premiums.")
        elif "Gender" in selected_question:
            st.info("Gender shows minimal impact on insurance charges. Other factors like age, BMI, and smoking status are far more influential.")
        elif "Children" in selected_question:
            st.info("Number of children has a moderate impact on insurance costs. More dependents can slightly increase overall family health costs, reflected in insurance premiums.")
        elif "Region" in selected_question:
            st.info("Different regions have varying cost structures due to factors like local healthcare costs, competition among providers, and state regulations.")
        elif "Feature Correlations" in selected_question:
            st.info("This heatmap shows how strongly each numeric feature is related to others. Higher absolute values (closer to 1 or -1) indicate stronger relationships.")

# Page 3: Prediction
elif page == "💰 Cost Prediction":
    st.title("💰 Predict Insurance Charges")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Enter Patient Details:")

        # Create input form for prediction
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
            # Encode categorical inputs to numerical values as expected by the model
            region_map = {'Northeast': 0, 'Southeast': 1, 'Southwest': 2, 'Northwest': 3}
            region_encoded = region_map[region]
            sex_encoded = 1 if sex == 'Male' else 0 # 0 for Female, 1 for Male
            smoker_encoded = 1 if smoker == 'Yes' else 0 # 0 for No, 1 for Yes

            # Create a DataFrame from the input data, matching the model's expected features order
            input_data = pd.DataFrame({
                'age': [age],
                'bmi': [bmi],
                'children': [children],
                'sex': [sex_encoded],
                'smoker': [smoker_encoded],
                'region': [region_encoded]
            })

            # Make prediction using the loaded model
            try:
                prediction = model.predict(input_data)[0]

                # Display the predicted insurance cost
                st.success(f"### 💰 Predicted Insurance Cost: ${prediction:,.2f}")

                # Show a summary of the input details for clarity
                st.subheader("📋 Input Summary:")
                summary_df = pd.DataFrame({
                    'Feature': ['Age', 'BMI', 'Children', 'Gender', 'Smoker', 'Region'],
                    'Value': [age, f"{bmi:.1f}", children, sex, smoker, region]
                })
                st.table(summary_df)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}. Please check inputs and ensure the model is correctly loaded.")

    with col2:
        st.markdown("### 💡 Prediction Tips")
        st.info("**Age**: Older individuals typically have higher insurance costs due to increased health risks.")
        st.info("**BMI**: A higher Body Mass Index (BMI) can indicate potential health issues, leading to increased costs.")
        st.warning("**Smoking**: This is often the biggest factor affecting insurance costs, as smoking is linked to numerous serious health conditions.")
        st.info("**Children**: Having more dependents (children) can sometimes increase overall family insurance costs.")
        st.info("**Region**: Geographic regions can have varying cost structures due to differences in healthcare infrastructure, regulations, and living costs.")

        # Show basic model information
        st.markdown("### 🤖 Model Information")
        st.success("**Model Type**: Random Forest Regressor - a powerful ensemble learning method for regression tasks.")
        st.success("**Features Used**: The model utilizes 6 key demographic and health factors to make predictions.")

# Footer section in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About This App")
st.sidebar.info("This app demonstrates machine learning for insurance cost prediction using demographic and health factors. It aims to provide insights into how different variables influence medical insurance premiums.")
