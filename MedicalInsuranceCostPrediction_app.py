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

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for Streamlit - this is crucial for deployment
import matplotlib
matplotlib.use('Agg')
plt.ioff()  # Turn off interactive mode

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
        # Create sample data for demonstration if model is not found
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
        try:
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)
        except:
            pass  # Might not have write permissions in deployment

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

# Debug: Print data info
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.write("Data shape:", df.shape)
    st.sidebar.write("Data types:", df.dtypes)
    st.sidebar.write("Sample data:")
    st.sidebar.write(df.head())
    st.sidebar.write("Unique values in smoker:", df['smoker'].unique())
    st.sidebar.write("Unique values in sex:", df['sex'].unique())

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

    # Improved visualization functions with better error handling
    def show_distribution_of_charges():
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['charges'], bins=30, color='teal', alpha=0.7, edgecolor='black')
            ax.set_title('Distribution of Medical Insurance Charges', fontsize=14, fontweight='bold')
            ax.set_xlabel('Charges ($)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error creating distribution chart: {str(e)}")
            return None

    def show_age_distribution():
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['age'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
            ax.set_title('Distribution of Age', fontsize=14, fontweight='bold')
            ax.set_xlabel('Age', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error creating age distribution: {str(e)}")
            return None

    def show_smoker_non_smoker_count():
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            smoker_counts = df['smoker'].value_counts()
            
            # Ensure we have both categories
            non_smoker_count = smoker_counts.get(0, 0)
            smoker_count = smoker_counts.get(1, 0)
            
            labels = ['Non-Smoker', 'Smoker']
            counts = [non_smoker_count, smoker_count]
            colors = ['lightblue', 'lightcoral']
            
            bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor='black')
            ax.set_title('Count of Smokers vs Non-Smokers', fontsize=14, fontweight='bold')
            ax.set_xlabel('Smoking Status', fontsize=12)
            ax.set_ylabel('Number of Individuals', fontsize=12)
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                           f'{count}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error creating smoker count chart: {str(e)}")
            return None

    def show_avg_bmi():
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            average_bmi = df['bmi'].mean()
            ax.hist(df['bmi'], bins=30, color='purple', alpha=0.7, edgecolor='black')
            ax.axvline(average_bmi, color='red', linestyle='--', linewidth=2,
                       label=f'Mean BMI: {average_bmi:.2f}')
            ax.set_title('Distribution of BMI', fontsize=14, fontweight='bold')
            ax.set_xlabel('BMI', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error creating BMI distribution: {str(e)}")
            return None

    def show_no_of_policyholders():
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            region_map = {0: 'Northeast', 1: 'Southeast', 2: 'Southwest', 3: 'Northwest'}
            region_counts = df['region'].value_counts()
            
            regions = []
            counts = []
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, region_name in region_map.items():
                count = region_counts.get(i, 0)
                regions.append(region_name)
                counts.append(count)
            
            bars = ax.bar(regions, counts, color=colors[:len(regions)], alpha=0.8, edgecolor='black')
            ax.set_title('Number of Policyholders by Region', fontsize=14, fontweight='bold')
            ax.set_xlabel('Region', fontsize=12)
            ax.set_ylabel('Number of Policyholders', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                           f'{count}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error creating region chart: {str(e)}")
            return None

    def show_charge_age():
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Filter data based on smoker status
            smokers = df[df['smoker'] == 1]
            non_smokers = df[df['smoker'] == 0]
            
            # Plot data
            if len(non_smokers) > 0:
                ax.scatter(non_smokers['age'], non_smokers['charges'], alpha=0.6, 
                          color='blue', label='Non-Smoker', s=50)
            if len(smokers) > 0:
                ax.scatter(smokers['age'], smokers['charges'], alpha=0.6, 
                          color='red', label='Smoker', s=50)
            
            ax.set_title('Charges vs. Age (Colored by Smoker Status)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Age', fontsize=12)
            ax.set_ylabel('Charges ($)', fontsize=12)
            if len(smokers) > 0 or len(non_smokers) > 0:
                ax.legend(title='Smoking Status')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error creating age vs charges chart: {str(e)}")
            return None

    def show_charges_smokervsnon():
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Get data for each group
            non_smoker_charges = df[df['smoker'] == 0]['charges']
            smoker_charges = df[df['smoker'] == 1]['charges']
            
            # Create box plot data
            data_to_plot = []
            labels = []
            colors = []
            
            if len(non_smoker_charges) > 0:
                data_to_plot.append(non_smoker_charges.values)
                labels.append('Non-Smoker')
                colors.append('lightblue')
            
            if len(smoker_charges) > 0:
                data_to_plot.append(smoker_charges.values)
                labels.append('Smoker')
                colors.append('lightcoral')
            
            if data_to_plot:
                box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_title('Medical Charges: Smokers vs Non-Smokers', fontsize=14, fontweight='bold')
            ax.set_xlabel('Smoking Status', fontsize=12)
            ax.set_ylabel('Charges ($)', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error creating smoker comparison chart: {str(e)}")
            return None

    def show_bmi_charge():
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Filter data based on smoker status
            smokers = df[df['smoker'] == 1]
            non_smokers = df[df['smoker'] == 0]
            
            # Plot data
            if len(non_smokers) > 0:
                ax.scatter(non_smokers['bmi'], non_smokers['charges'], alpha=0.6, 
                          color='blue', label='Non-Smoker', s=50)
            if len(smokers) > 0:
                ax.scatter(smokers['bmi'], smokers['charges'], alpha=0.6, 
                          color='red', label='Smoker', s=50)
            
            ax.set_title('Charges vs. BMI (Colored by Smoker Status)', fontsize=14, fontweight='bold')
            ax.set_xlabel('BMI', fontsize=12)
            ax.set_ylabel('Charges ($)', fontsize=12)
            if len(smokers) > 0 or len(non_smokers) > 0:
                ax.legend(title='Smoking Status')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error creating BMI vs charges chart: {str(e)}")
            return None

    def show_men_women_charge():
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Get data for each gender (0=Female, 1=Male in your encoding)
            female_charges = df[df['sex'] == 0]['charges']
            male_charges = df[df['sex'] == 1]['charges']
            
            # Create box plot data
            data_to_plot = []
            labels = []
            colors = []
            
            if len(female_charges) > 0:
                data_to_plot.append(female_charges.values)
                labels.append('Female')
                colors.append('pink')
            
            if len(male_charges) > 0:
                data_to_plot.append(male_charges.values)
                labels.append('Male')
                colors.append('lightblue')
            
            if data_to_plot:
                box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_title('Medical Charges: Male vs Female', fontsize=14, fontweight='bold')
            ax.set_xlabel('Gender', fontsize=12)
            ax.set_ylabel('Charges ($)', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error creating gender comparison chart: {str(e)}")
            return None

    def show_correlation_children_charge():
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            children_avg = df.groupby('children')['charges'].mean()
            
            # Get the range of children values
            min_children = df['children'].min()
            max_children = df['children'].max()
            children_range = range(int(min_children), int(max_children) + 1)
            
            avg_charges = []
            actual_children = []
            
            for i in children_range:
                if i in children_avg.index:
                    avg_charges.append(children_avg[i])
                    actual_children.append(i)
            
            if avg_charges:
                colors = plt.cm.viridis(np.linspace(0, 1, len(actual_children)))
                bars = ax.bar(actual_children, avg_charges, color=colors, alpha=0.8, edgecolor='black')
                
                # Add value labels on bars
                for bar, avg_charge in zip(bars, avg_charges):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(avg_charges) * 0.01,
                           f'${avg_charge:,.0f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title('Average Charges by Number of Children', fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Children', fontsize=12)
            ax.set_ylabel('Average Charges ($)', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error creating children correlation chart: {str(e)}")
            return None

    def show_numeric_features():
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            numeric_cols = ['age', 'bmi', 'children', 'charges']
            
            # Ensure all columns exist
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(available_cols) >= 2:
                corr_matrix = df[available_cols].corr()
                
                # Create heatmap manually
                im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
                
                # Set ticks and labels
                ax.set_xticks(range(len(available_cols)))
                ax.set_yticks(range(len(available_cols)))
                ax.set_xticklabels(available_cols, rotation=45)
                ax.set_yticklabels(available_cols)
                
                # Add correlation values as text
                for i in range(len(available_cols)):
                    for j in range(len(available_cols)):
                        text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
                        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                               ha='center', va='center', color=text_color, fontsize=10)
                
                plt.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title('Correlation Between Numeric Features', fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Insufficient numeric columns for correlation', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error creating correlation matrix: {str(e)}")
            return None

    # Questions dictionary
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

    # Create two columns
    col1, col2 = st.columns([3, 1])

    with col1:
        # Execute selected visualization and display
        try:
            fig = questions[selected_question]()
            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)  # Important: close figure to free memory
            else:
                st.error("Could not generate the selected visualization.")
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            # Debug information
            with st.expander("Debug Information"):
                st.write(f"Data shape: {df.shape}")
                st.write(f"Data columns: {df.columns.tolist()}")
                st.write(f"Data types: {df.dtypes}")
                st.write("First few rows:")
                st.write(df.head())

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

                # Additional insights
                if smoker == 'Yes':
                    st.warning("⚠️ Smoking significantly increases insurance costs!")
                if bmi > 30:
                    st.info("ℹ️ High BMI may contribute to increased costs")
                if age > 50:
                    st.info("ℹ️ Age is a significant factor in insurance pricing")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Debug info:")
                st.write(f"Input data shape: {input_data.shape}")
                st.write(f"Input data: {input_data}")

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
        
        # Show data statistics
        st.markdown("### 📊 Dataset Stats")
        st.metric("Total Records", len(df))
        st.metric("Average Cost", f"${df['charges'].mean():,.0f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About This App")
st.sidebar.info("This app demonstrates machine learning for insurance cost prediction using demographic and health factors.")

# Add requirements.txt information
st.sidebar.markdown("### 📋 Requirements")
with st.sidebar.expander("View Requirements"):
    st.code("""
streamlit
pandas
matplotlib
seaborn
numpy
scikit-learn
    """)
