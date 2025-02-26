import streamlit as st
import pandas as pd
import zipfile
import joblib
import re

st.title("Problem Prediction App")

# --- Unzip Model ---
try:
    with zipfile.ZipFile("problem_prediction_model.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    st.success("Model unzipped successfully.")
except FileNotFoundError:
    st.error("Model zip file not found.")
    st.stop()
except Exception as e:
    st.error(f"Error unzipping model: {e}")
    st.stop()

# --- Load Model ---
try:
    model_data = joblib.load("problem_prediction_model.pkl")
    clf = model_data["classifier"]
    vectorizer = model_data["vectorizer"]
    mlb = model_data["mlb"]
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Prediction ---
uploaded_file = st.file_uploader("Upload CSV file for problem prediction", type=["csv"])

if uploaded_file:
    if st.button("Predict Problems"):
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            X_test = vectorizer.transform(uploaded_df["Cleaned_Text"])
            y_pred = clf.predict(X_test)
            predicted_problems = mlb.inverse_transform(y_pred)
            uploaded_df["Predicted Problems"] = predicted_problems

            st.subheader("Predicted Problems:")
            for index, row in uploaded_df.iterrows():
                st.write(f"**Text:** {row['Text']}")
                st.write(f"**Cleaned Text:** {row['Cleaned_Text']}")
                st.write(f"**Predicted Problems:** {row['Predicted Problems']}")
                st.write("---")

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
