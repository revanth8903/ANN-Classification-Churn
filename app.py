import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd
import numpy as np

## Load the trained model
model = tf.keras.models.load_model('model.h5')

## Load the Encoders and Scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit app
st.title("Customer Churn Prediction")

## User Input
Geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', value=0.0)
credit_score = st.number_input('Credit Score', value=600.0)
estimated_salary = st.number_input('Estimated Salary', value=50000.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

## Prepare the Input Data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out())

## Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Ensure correct feature order before scaling
expected_columns = scaler.feature_names_in_  # Get expected column order from scaler
input_data = input_data[expected_columns]  # Reorder columns

## Scale the input data
input_data_scaled = scaler.transform(input_data)

## Predict Churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

## Display results
st.write("### 🔍 Prediction Probability:", prediction_proba)

if prediction_proba > 0.5:
    st.write('🛑 The Customer is **likely** to churn.')
else:
    st.write('✅ The Customer is **not likely** to churn.')

