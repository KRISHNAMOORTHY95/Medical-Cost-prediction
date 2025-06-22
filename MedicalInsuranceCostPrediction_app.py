import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("/content/model.pkl")

df = pd.read_csv("/content/medical_insurance.csv")  # Load your dataset

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Introduction", "Visualizations", "Cost Prediction"])

# Page 1: Project Introduction

if page == "Project Introduction":
    st.title("Medical Insurance Cost Prediction")
    st.markdown("""
    This project uses a dataset of medical insurance records to analyze key factors affecting insurance cost.

    **Goals:**
    - Explore patterns in charges based on age, BMI, smoking habits, region, etc.
    - Use Support Vector Regression (SVR) to predict insurance charges.
    """)

# Page 2: Visualizations

elif page == "Visualizations":
    st.title("Exploratory Data Analysis")

    def distribution_of_charges():
        plt.figure(figsize=(10, 5))
        sns.histplot(df['charges'], kde=True, bins=30, color='teal')
        plt.title('Distribution of Medical Insurance Charges')
        plt.xlabel('Charges')
        plt.ylabel('Frequency')
        plt.tight_layout()

    def age_distribution():
        plt.figure(figsize=(10, 5))
        sns.histplot(df['age'], kde=True, bins=20, color='skyblue')
        plt.title('Distribution of Age')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.tight_layout()

    def smoker_non_smoker():
        plt.figure(figsize=(6, 4))
        sns.countplot(x='smoker', data=df, palette='Set2')
        plt.title('Count of Smokers vs Non-Smokers')
        plt.xlabel('Smoking Status')
        plt.ylabel('Number of Individuals')
        plt.tight_layout()

    def avg_bmi():
        average_bmi = df['bmi'].mean()
        plt.figure(figsize=(10, 5))
        sns.histplot(df['bmi'], kde=True, bins=30, color='purple')
        plt.axvline(average_bmi, color='red', linestyle='--', label=f'Mean BMI: {average_bmi:.2f}')
        plt.title('Distribution of BMI')
        plt.xlabel('BMI')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()

    def no_of_policyholders():
        region_counts = df['region'].value_counts()
        plt.figure(figsize=(8, 5))
        sns.countplot(x='region', data=df, palette='Set3', order=region_counts.index)
        plt.title('Number of Policyholders by Region')
        plt.xlabel('Region')
        plt.ylabel('Number of Policyholders')
        plt.tight_layout()

    def charge_age():
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='age', y='charges', hue='smoker', data=df, alpha=0.6, palette='Set1')
        plt.title('Charges vs. Age Colored by Smoker Status')
        plt.xlabel('Age')
        plt.ylabel('Charges')
        plt.legend(title='Smoker')
        plt.tight_layout()

    def charges_smokervsnon():
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='smoker', y='charges', data=df, palette='Set2')
        plt.title('Medical Charges: Smokers vs Non-Smokers')
        plt.xlabel('Smoking Status')
        plt.ylabel('Charges')
        plt.tight_layout()

    def bmi_charge():
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df, alpha=0.6, palette='Set1')
        plt.title('Charges vs. BMI (Colored by Smoker Status)')
        plt.xlabel('BMI')
        plt.ylabel('Charges')
        plt.legend(title='Smoker')
        plt.tight_layout()

    def men_women_charge():
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='sex', y='charges', data=df, palette='pastel')
        plt.title('Medical Charges: Male vs Female')
        plt.xlabel('Gender')
        plt.ylabel('Charges')
        plt.tight_layout()

    def correlation_children_charge():
        plt.figure(figsize=(8, 5))
        sns.barplot(x='children', y='charges', data=df, estimator='mean', palette='coolwarm')
        plt.title('Average Charges by Number of Children')
        plt.xlabel('Number of Children')
        plt.ylabel('Average Charges')
        plt.tight_layout()

    def smoke_medicalcharge():
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='age', y='charges', hue='smoker', palette='Set1', alpha=0.6)
        plt.title('Impact of Smoking and Age on Medical Charges')
        plt.xlabel('Age')
        plt.ylabel('Charges')
        plt.legend(title='Smoker')
        plt.tight_layout()

    def impact_gender():
        smokers_df = df[df['smoker'] == 1]
        smokers_df['sex_label'] = smokers_df['sex'].map({0: 'Female', 1: 'Male'})
        smokers_df['region_label'] = smokers_df['region'].map({0: 'northeast', 1: 'southeast', 2: 'southwest', 3: 'northwest'})
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=smokers_df, x='region_label', y='charges', hue='sex_label', palette='pastel')
        plt.title('Charges by Region and Gender (Smokers Only)')
        plt.xlabel('Region')
        plt.ylabel('Charges')
        plt.legend(title='Gender')
        plt.tight_layout()

    def age_BMI_smoking():
        df['smoker_label'] = df['smoker'].map({0: 'No', 1: 'Yes'})
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df,
                        x='age', y='charges',
                        hue='smoker_label',
                        size='bmi',
                        sizes=(20, 200),
                        alpha=0.7,
                        palette='Set1')
        plt.title('Charges by Age, BMI (Size), and Smoking Status')
        plt.xlabel('Age')
        plt.ylabel('Insurance Charges')
        plt.legend(title='Smoker Status')
        plt.tight_layout()

    def obese_smokers():
        group1 = df[(df['smoker'] == 1) & (df['bmi'] > 30)]
        group2 = df[(df['smoker'] == 0) & (df['bmi'] <= 30)]
        group1['group'] = 'Obese Smokers'
        group2['group'] = 'Non-Obese Non-Smokers'
        combined_df = pd.concat([group1, group2])
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=combined_df, x='group', y='charges', palette='coolwarm')
        plt.title('Insurance Charges: Obese Smokers vs Non-Obese Non-Smokers')
        plt.ylabel('Charges')
        plt.tight_layout()

    def individuals_paying():
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x='charges', color='orange')
        plt.title('Boxplot of Insurance Charges')
        plt.xlabel('Charges')
        plt.tight_layout()

    def extreme_BMI():
        plt.figure(figsize=(8, 4))
        sns.histplot(df['bmi'], bins=30, kde=True, color='teal')
        plt.title('Distribution of BMI')
        plt.xlabel('BMI')
        plt.tight_layout()

    def numeric_features():
        numeric_cols = ['age', 'bmi', 'children', 'charges']
        corr_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Between Numeric Features')
        plt.tight_layout()

    def target_variable():
        correlations = df.corr(numeric_only=True)['charges'].drop('charges').sort_values(key=abs, ascending=True)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=correlations.values, y=correlations.index, palette='viridis')
        plt.title('Correlation of Features with Insurance Charges')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Features')
        plt.axvline(0, color='gray', linestyle='--')
        plt.tight_layout()


    questions = {
        "1. Distribution of Charges": distribution_of_charges,
        "2. Age Distribution": age_distribution,
        "3. Smokers vs Non-Smokers": smoker_non_smoker,
        "4. Average BMI": avg_bmi,
        "5. Policyholders by Region": no_of_policyholders,
        "6. Charges vs Age": charge_age,
        "7. Charges: Smokers vs Non-Smokers": charges_smokervsnon,
        "8. Charges vs BMI": bmi_charge,
        "9. Charges by Gender": men_women_charge,
        "10. Charges vs Number of Children": correlation_children_charge,
        "11. Charges by Age & Smoking": smoke_medicalcharge,
        "12. Charges by Region & Gender (Smokers)": impact_gender,
        "13. Combined Effect: Age, BMI, Smoking": age_BMI_smoking,
        "14. Obese Smokers vs Non-Obese Non-Smokers": obese_smokers,
        "15. Outliers in Charges": individuals_paying,
        "16. Outliers in BMI": extreme_BMI,
        "17. Correlation of Numeric Features": numeric_features,
        "18. Most Important Features for Charges": target_variable,
    }

    selected_question = st.selectbox("Select a question to visualize:", list(questions.keys()))
    questions[selected_question]()
    fig = plt.gcf()
    st.pyplot(fig)


# Page 3: Prediction

elif page == "Cost Prediction":
    st.title(" Predict Insurance Charges (Random Forest Model)")
    st.subheader(" Enter the following details:")

    # User Inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=1)
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    region_input = st.selectbox("Region", ['northeast', 'southeast', 'southwest', 'northwest'])
    sex = st.selectbox("Sex", ["male", "female"])

    # Manual Encoding
    region_map = {'northeast': 0, 'southeast': 1, 'southwest': 2, 'northwest': 3}
    region_encoded = region_map[region_input]
    sex_encoded = 1 if sex == 'male' else 0
    smoker_encoded = 1 if smoker == 'Yes' else 0

    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex': [sex_encoded],
        'smoker': [smoker_encoded],
        'region': [region_encoded]
    })

    st.write(" Input Data Used for Prediction:")
    st.dataframe(input_data)

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.success(f" Predicted Insurance Cost: {prediction:,.2f}")

