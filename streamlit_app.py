import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model
try:
    classifier = joblib.load("my_model.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load matched_form_data.csv and retrain TF-IDF
try:
    df = pd.read_csv("matched_form_data.csv")
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df['Cleaned_Text'])  # Train vectorizer on Cleaned_Text
except Exception as e:
    st.error(f"Failed to load CSV or retrain vectorizer: {e}")
    st.stop()

st.title("Model Analysis App")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    if st.button("Run Analysis"):
        try:
            text = uploaded_file.read().decode("utf-8")
            text_vectorized = vectorizer.transform([text])  # Vectorize input text
            prediction = classifier.predict(text_vectorized)[0]

            st.write(f"Prediction: {prediction}")

        except Exception as e:
            st.error(f"Error processing text: {e}")
