import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load(r'C:\Users\sarat\MachineLearning\predictibe model\model.pkl')

# Create input fields for user input
st.title('Iris Flower Predictor')
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Create a button for prediction
if st.button('Predict'):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.write(f'Predicted class: {prediction[0]}')  # Adjust this if necessary based on your output
