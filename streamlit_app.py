import streamlit as st
import joblib
import pandas as pd

# Load the model
try:
    classifier = joblib.load("my_model.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("Model Analysis App")

uploaded_file = st.file_uploader("Upload Cleaned CSV File", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        features = df.drop('target', axis=1) #Drop the target column.
        predictions = classifier.predict(features)
        df['prediction'] = predictions

        st.write("Analysis Results:")
        st.dataframe(df)

        st.write("Prediction Value Counts:")
        st.write(df['prediction'].value_counts())

    except Exception as e:
        st.error(f"Error processing CSV: {e}")
