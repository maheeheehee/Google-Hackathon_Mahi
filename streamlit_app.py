import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the model
try:
    classifier = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load CSVs and retrain TF-IDF and PCA
try:
    tech_text_df = pd.read_csv("merged_dataset.csv", dtype=str, low_memory=False)
    form_df = pd.read_csv("processed_text_data.csv", dtype=str, low_memory=False)

    form_df["Cleaned_Text"] = form_df["Cleaned_Text"].fillna("")
    tech_text_df["PROBLEM_TYPE"] = tech_text_df["PROBLEM_TYPE"].fillna("")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(list(form_df["Cleaned_Text"]) + list(tech_text_df["PROBLEM_TYPE"]))

    form_tfidf = tfidf_matrix[: len(form_df)]

    scaler = StandardScaler()
    tfidf_scaled = scaler.fit_transform(form_tfidf.toarray())

    pca = PCA(n_components=0.95)
    tfidf_pca = pca.fit_transform(tfidf_scaled)

except Exception as e:
    st.error(f"Failed to load CSVs or retrain: {e}")
    st.stop()

st.title("Enterprise Process Automation")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    # Preprocess and transform the input text
    text_vectorized = vectorizer.transform([text])
    text_scaled = scaler.transform(text_vectorized.toarray())
    text_pca = pca.transform(text_scaled)

    # Make prediction
    prediction = classifier.predict(text_pca)[0]

    st.write(f"Category: {prediction}")
