import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set the title of the app
st.title('🧠 Prédiction Alzheimer - Analyse Complète')

# Load the model with error handling
try:
    model = joblib.load("alzheimer.joblib")
except FileNotFoundError:
    st.error("❌ Erreur : Fichier 'alzheimer.joblib' introuvable. Veuillez le placer dans le même répertoire.")
    st.stop()

# Define the input fields for the user to enter their data
col1, col2, col3 = st.columns(3)

# Column 1: Demographic data
with col1:
    st.subheader("📊 Données Démographiques")
    age = st.slider("Âge", 50, 100, 70)
    gender = st.radio("Genre", ['Homme', 'Femme'])
    education = st.selectbox("Niveau d'Éducation", ['Primaire', 'Secondaire', 'Universitaire'])
    marital_status = st.selectbox("Statut Matrimonial", ['Marié(e)', 'Célibataire', 'Divorcé(e)', 'Veuf/Veuve'])

# Column 2: Medical data
with col2:
    st.subheader("🩺 Données Médicales")
    bmi = st.slider("IMC", 15.0, 40.0, 25.0)
    cognitive_score = st.slider("Score Cognitif (MMSE)", 0, 30, 25)
    apoe = st.radio("Gène APOE-ε4", ['Présent', 'Absent'])
    hypertension = st.checkbox("Hypertension")
    diabetes = st.checkbox("Diabète")

# Column 3: Behavioral factors
with col3:
    st.subheader("🧠 Facteurs Comportementaux")
    physical_activity = st.select_slider("Activité Physique", ['Nulle', 'Légère', 'Modérée', 'Intense'])
    diet = st.radio("Régime Alimentaire", ['Méditerranéen', 'Occidental', 'Végétarien'])
    smoking = st.selectbox("Tabagisme", ['Non-fumeur', 'Ancien fumeur', 'Fumeur actuel'])
    alcohol = st.select_slider("Consommation d'Alcool", ['Jamais', 'Occasionnelle', 'Régulière'])

# Prepare the data for prediction
data = pd.DataFrame([{
    'Age': age,
    'BMI': bmi,
    'MMSE': cognitive_score,
    'Gender_Female': 1 if gender == 'Femme' else 0,
    'Gender_Male': 1 if gender == 'Homme' else 0,
    'Education_Primary': 1 if education == 'Primaire' else 0,
    'Education_Secondary': 1 if education == 'Secondaire' else 0,
    'Education_University': 1 if education == 'Universitaire' else 0,
    'Married': 1 if marital_status == 'Marié(e)' else 0,
    'Single': 1 if marital_status == 'Célibataire' else 0,
    'Divorced': 1 if marital_status == 'Divorcé(e)' else 0,
    'Widowed': 1 if marital_status == 'Veuf/Veuve' else 0,
    'APOE4_Present': 1 if apoe == 'Présent' else 0,
    'APOE4_Absent': 1 if apoe == 'Absent' else 0,
    'Hypertension': 1 if hypertension else 0,
    'Diabetes': 1 if diabetes else 0,
    'Physical_Activity_None': 1 if physical_activity == 'Nulle' else 0,
    'Physical_Activity_Light': 1 if physical_activity == 'Légère' else 0,
    'Physical_Activity_Moderate': 1 if physical_activity == 'Modérée' else 0,
    'Physical_Activity_Intense': 1 if physical_activity == 'Intense' else 0,
    'Diet_Mediterranean': 1 if diet == 'Méditerranéen' else 0,
    'Diet_Western': 1 if diet == 'Occidental' else 0,
    'Diet_Vegetarian': 1 if diet == 'Végétarien' else 0,
    'Smoking_Current': 1 if smoking == 'Fumeur actuel' else 0,
    'Smoking_Former': 1 if smoking == 'Ancien fumeur' else 0,
    'Smoking_Never': 1 if smoking == 'Non-fumeur' else 0,
    'Alcohol_Never': 1 if alcohol == 'Jamais' else 0,
    'Alcohol_Occasional': 1 if alcohol == 'Occasionnelle' else 0,
    'Alcohol_Regular': 1 if alcohol == 'Régulière' else 0
}])

# Ensure that the data has the same columns as the model was trained on
model_columns = model.feature_names_in_  # Getting feature names from the model

# Check if any columns are missing from the input data and add them with default value 0
for col in model_columns:
    if col not in data.columns:
        data[col] = 0

# Reorder the columns to match the model's expected order
data = data[model_columns]

# Predict when the button is pressed
if st.button("🔍 Analyser le Risque"):
    st.write("Traitement en cours...")

    # Prediction
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0, 1]  # Get probability of Alzheimer's risk

    # Display results
    if prediction == 1:
        st.error(f"⚠️ Risque Élevé de Maladie d'Alzheimer ({probability:.1%})")
    else:
        st.success(f"✅ Risque Faible ({(1 - probability):.1%} de sécurité)")

            
        
       










