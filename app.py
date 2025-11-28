import streamlit as st
import joblib
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        # Try joblib first
        return joblib.load("house_model.pkl")
    except (KeyError, EOFError, pickle.UnpicklingError) as e:
        st.error(f"❌ Model file corrupted: {str(e)}")
        st.warning("""
        **Your model file is corrupted!**
        
        Please re-export your model from Google Colab:
        
        ```python
        import joblib
        import pickle
        
        # Save with pickle protocol 4
        with open("house_model.pkl", "wb") as f:
            pickle.dump(your_model, f, protocol=4)
        
        # Then download from Colab
        from google.colab import files
        files.download("house_model.pkl")
        ```
        
        Then upload the new file to replace the current one.
        """)
        return None

model = load_model()

# Title and description
st.title("🏠 House Price Prediction")
st.write("Enter the house features below to predict its price")

# Create input columns
col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input("Latitude", value=37.88, min_value=-90.0, max_value=90.0)
    longitude = st.number_input("Longitude", value=-122.23, min_value=-180.0, max_value=180.0)
    rooms = st.number_input("Number of Rooms", value=6.98, min_value=1.0, max_value=10.0, step=0.1)
    bedrooms = st.number_input("Bedrooms", value=1.02, min_value=1.0, max_value=10.0, step=0.1)

with col2:
    population = st.number_input("Population", value=322, min_value=1, max_value=10000, step=1)
    households = st.number_input("Households", value=2.55, min_value=1.0, max_value=100.0, step=0.1)
    median_income = st.number_input("Median Income", value=8.3252, min_value=0.0, max_value=15.0, step=0.1)
    house_age = st.number_input("House Age (years)", value=41.0, min_value=1, max_value=52, step=1)

# Prepare input data
input_data = np.array([
    median_income,
    house_age,
    rooms,
    bedrooms,
    population,
    households,
    latitude,
    longitude
]).reshape(1, -1)

# Make prediction
if st.button("🎯 Predict Price", use_container_width=True):
    if model is None:
        st.error("❌ Model not loaded. Please fix the model file first.")
    else:
        try:
            prediction = model.predict(input_data)
            price = prediction[0]
            
            # Display result
            st.success("✅ Prediction Complete!")
            st.metric(
                label="Predicted House Price",
                value=f"${price:,.2f}",
                delta=None
            )
            
            # Display input summary
            with st.expander("📋 Input Summary"):
                st.write(f"**Location:** ({latitude}, {longitude})")
                st.write(f"**Rooms:** {rooms} | **Bedrooms:** {bedrooms}")
                st.write(f"**Population:** {population} | **Households:** {households}")
                st.write(f"**Median Income:** ${median_income}k | **House Age:** {house_age} years")
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

# Footer
st.divider()
st.caption("Built with Streamlit | Deployed on Render")
