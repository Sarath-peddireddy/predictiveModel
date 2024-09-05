import streamlit as st
import joblib
import numpy as np

# Load your trained model with caching
@st.cache_resource
def load_model():
    return joblib.load(r'predictibe model/model.pkl')  # Ensure the path is correct

model = load_model()

# Create input fields for user input
st.title('Iris Flower Predictor')

# Set up input fields with sensible defaults and ranges
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=5.0, value=0.2, step=0.1)

# Create a button for prediction
if st.button('Predict'):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display prediction
    st.write(f'Predicted class: {prediction[0]}')  # Adjust this if necessary based on your output
