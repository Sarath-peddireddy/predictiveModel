import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.datasets import load_iris

# Load the model with caching
@st.cache_resource
def load_model():
    model_path = 'model.pkl'  # Ensure this path matches where your model is saved
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        st.stop()

# Load the prediction model
model = load_model()

# Load the Iris dataset
iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['species'] = iris.target_names[iris.target]

# App Title
st.title('Iris Species Predictor')

# Input features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

# Button to predict
if st.button('Predict'):
    try:
        # Validate the input; ensure it is a valid 2D array shape for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Ensure input is valid
        if input_data.shape[1] != 4 or np.any(input_data < 0):
            st.error("Input values are incorrect. Please enter valid positive values for all features.")
        else:
            # Make prediction
            prediction = model.predict(input_data)[0]
            st.success(f'Predicted species: {iris.target_names[prediction]}')

            # Optionally, display probabilities
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)
                st.write("Probabilities for each class:")
                for i, species in enumerate(iris.target_names):
                    st.write(f"{species}: {probabilities[0][i]:.2f}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Displaying the existing Iris dataset as a scatter plot
fig = px.scatter(iris_data, x='sepal length (cm)', y='sepal width (cm)', color='species', title='Iris Dataset Scatter Plot')
st.plotly_chart(fig)

# Introduction/Help Section
st.sidebar.header("About This App")
st.sidebar.write("This app predicts the species of the Iris flower using a trained machine learning model.")
st.sidebar.write("Enter the sepal and petal dimensions to get a prediction.")
