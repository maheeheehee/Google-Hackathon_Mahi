import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model
try:
    classifier = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load the CSVs and retrain the TF-IDF vectorizer
try:
    # Replace 'processed.csv' and 'merged.csv' with your actual filenames
    processed_df = pd.read_csv("processed.csv")
    merged_df = pd.read_csv("merged.csv")

    # Assuming your text data is in a column named 'text'
    training_texts = pd.concat([processed_df['text'], merged_df['text']])

    vectorizer = TfidfVectorizer()
    vectorizer.fit(training_texts)  # Retrain the vectorizer

except Exception as e:
    st.error(f"Failed to load CSVs or retrain vectorizer: {e}")
    st.stop()

st.title("Enterprise Process Automation")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    # Preprocess the text using the recreated TF-IDF vectorizer
    text_vectorized = vectorizer.transform([text])

    # Make prediction
    prediction = classifier.predict(text_vectorized)[0]

    st.write(f"Category: {prediction}")
