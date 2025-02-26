import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("Model Analysis App")

# Load the model and vectorizer together
try:
    model_data = joblib.load("my_model.pkl")
    # Check if model_data is a dict with both classifier and vectorizer
    if isinstance(model_data, dict) and 'classifier' in model_data and 'vectorizer' in model_data:
        classifier = model_data['classifier']
        vectorizer = model_data['vectorizer']
        st.success("Model and vectorizer loaded successfully")
    else:
        classifier = model_data  # Assume it's just the model
        st.warning("Only model loaded, vectorizer will be trained from CSV")
        # We'll need to load the vectorizer separately or retrain it
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Only load the reference data if we need to train a vectorizer
if 'vectorizer' not in locals():
    try:
        df = pd.read_csv("matched_form_data.csv")
        vectorizer = TfidfVectorizer()
        vectorizer.fit(df['Cleaned_Text'])
        st.success("Vectorizer trained from CSV data")
    except Exception as e:
        st.error(f"Failed to load CSV or train vectorizer: {e}")
        st.stop()

uploaded_file = st.file_uploader("Upload matched_form_data.csv", type=["csv"])
if uploaded_file:
    if st.button("Run Analysis"):
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            
            # Ensure the Cleaned_Text column exists
            if 'Cleaned_Text' not in uploaded_df.columns:
                st.error("The uploaded CSV must contain a 'Cleaned_Text' column")
                st.stop()
                
            # Transform the text using the same vectorizer that was used during training
            text_vectorized = vectorizer.transform(uploaded_df['Cleaned_Text'])
            
            # Display feature count info for debugging
            st.info(f"Input features: {text_vectorized.shape[1]}, Model expects: {classifier.n_features_in_ if hasattr(classifier, 'n_features_in_') else 'unknown'}")
            
            # Make predictions
            predictions = classifier.predict(text_vectorized)
            uploaded_df['prediction'] = predictions
            
            st.write("Analysis Results:")
            st.dataframe(uploaded_df)
            st.write("Prediction Value Counts:")
            st.write(uploaded_df['prediction'].value_counts())
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
