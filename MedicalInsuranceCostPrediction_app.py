import streamlit as st
import pandas as pd
import numpy as np
import joblib

model=joblib.load("C:/Users/vicky/OneDrive/Desktop/MiniProject/MiniProject3/MiniProject3Sourcedocuments/randomforest_insurance_model.pkl")

st.title("Medical Insurance Cost Predictor")


# Sidebar input
age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 1)
sex = st.sidebar.selectbox("Sex", ['male', 'female'])
smoker = st.sidebar.selectbox("Smoker", ['yes', 'no'])
region = st.sidebar.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])


input_data = {
    'age': age,
    'bmi': bmi,
    'children': children,
    'sex_female': 1 if sex=='female' else 0,
    'sex_male': 1 if sex == 'male' else 0,
     'smoker_no':   1 if smoker =='no' else 0,
    'smoker_yes': 1 if smoker == 'yes' else 0,
    'region_northeast':1 if region=='northeast' else 0,
    'region_northwest': 1 if region == 'northwest' else 0,
    'region_southeast': 1 if region == 'southeast' else 0,
    'region_southwest': 1 if region == 'southwest' else 0
}

input_df = pd.DataFrame([input_data])

prediction = model.predict(input_df)[0]
st.subheader(f"Predicted Insurance Cost: ₹ {prediction:,.2f}")

# Optional: Feature importance visualization
if hasattr(model, "feature_importances_"):
    st.subheader("Feature Importance")
    import matplotlib.pyplot as plt
    import seaborn as sns

    feat_importances = pd.Series(model.feature_importances_, index=input_df.columns)
    fig, ax = plt.subplots()
    sns.barplot(x=feat_importances.values, y=feat_importances.index, ax=ax)
    st.pyplot(fig)
