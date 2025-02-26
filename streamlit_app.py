import streamlit as st
import joblib
import pandas as pd

st.title("Model Analysis App")

# Load the model and vectorizer together
try:
    model_data = joblib.load("my_model_new1.pkl")  # Use the new model file
    classifier = model_data['classifier']
    vectorizer = model_data['vectorizer']
    st.success("Model and vectorizer loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload CSV file for analysis", type=["csv"])
if uploaded_file:
    if st.button("Run Analysis"):
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            text_vectorized = vectorizer.transform(uploaded_df['Cleaned_Text'])
            predictions = classifier.predict(text_vectorized)
            uploaded_df['prediction'] = predictions
            
            st.write("Analysis Results:")
            st.dataframe(uploaded_df)
            st.write("Prediction Value Counts:")
            st.write(uploaded_df['prediction'].value_counts())
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
