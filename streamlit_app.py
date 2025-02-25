import streamlit as st
import pandas as pd
import joblib
import spacy
from transformers import pipeline
import os

# Load spaCy model (from the directory where setup.sh installed it)
model_name = "en_core_web_sm"
model_path = os.path.join(os.getcwd(), model_name)

try:
    ner_model = spacy.load(model_path)
except OSError:
    st.error(f"spaCy model '{model_name}' could not be loaded from '{model_path}'.")
    st.stop()

# Load NLP models
sentiment_analyzer = pipeline("sentiment-analysis")  # Sentiment Analysis

# Load text classification model
try:
    classifier = joblib.load("my_model.pkl")  # Replace "model.pkl" if needed
except Exception as e:
    st.error(f"Failed to load text classifier model: {e}")
    st.stop()

# Streamlit UI
st.title("AI-Powered Text Processing App")
st.write("Upload documents for OCR, classification, and analysis.")

# File Upload
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    st.subheader("Extracted Text:")
    st.text_area("File Content", text, height=200)

    # Named Entity Recognition (NER)
    doc = ner_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    st.subheader("Named Entities:")
    st.write(pd.DataFrame(entities, columns=["Entity", "Category"]))

    # Sentiment Analysis
    sentiment = sentiment_analyzer(text[:512])  # Limit to 512 characters
    st.subheader("Sentiment Analysis:")
    st.write(sentiment)

    # Text Classification
    try:
        prediction = classifier.predict([text])[0]
        st.subheader("Text Classification:")
        st.write(f"Predicted Category: **{prediction}**")
    except Exception as e:
        st.error(f"Error during text classification: {e}")
