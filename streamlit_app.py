import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model (RandomForestClassifier)
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

uploaded_file = st.file_uploader("Upload matched_form_data.csv", type=["csv"])

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
