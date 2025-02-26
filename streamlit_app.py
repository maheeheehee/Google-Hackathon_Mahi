import streamlit as st
import joblib
import pandas as pd

# Load the model
try:
    classifier = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("Model Analysis App")

uploaded_file = st.file_uploader("Upload matched_form_data.csv", type=["csv"])

if uploaded_file:
    if st.button("Run Analysis"):
        try:
            df = pd.read_csv(uploaded_file)

            # Separate features (TF-IDF columns) from the target
            tfidf_columns = [col for col in df.columns if col not in ['Text', 'Label', 'Cleaned_Text', 'Matched_Problems']]
            features = df[tfidf_columns]

            predictions = classifier.predict(features)
            df['prediction'] = predictions

            st.write("Analysis Results:")
            st.dataframe(df)

            st.write("Prediction Value Counts:")
            st.write(df['prediction'].value_counts())

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
