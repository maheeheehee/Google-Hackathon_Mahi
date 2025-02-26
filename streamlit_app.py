import streamlit as st
import joblib
import pandas as pd
import sklearn

st.title("Model Analysis App")
st.write(f"Using scikit-learn version: {sklearn.__version__}")

# First let's try to check if this is a pickle compatibility issue
try:
    model_data = joblib.load("my_model.pkl")
    st.success("Model loaded successfully")
    
    # Check if model_data is a dict with both classifier and vectorizer
    if isinstance(model_data, dict) and 'classifier' in model_data and 'vectorizer' in model_data:
        classifier = model_data['classifier']
        vectorizer = model_data['vectorizer']
        st.success("Model and vectorizer extracted successfully")
    else:
        classifier = model_data  # Assume it's just the model
        # We'll need to load the vectorizer separately
        try:
            df = pd.read_csv("matched_form_data.csv")
            vectorizer = TfidfVectorizer()
            vectorizer.fit(df['Cleaned_Text'])
            st.success("Vectorizer trained from CSV data")
        except Exception as e:
            st.error(f"Failed to load CSV or train vectorizer: {e}")
            st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.error("This is likely a scikit-learn version mismatch. Please ensure you're using the same version of scikit-learn that was used to train the model.")
    st.stop()

uploaded_file = st.file_uploader("Upload matched_form_data.csv", type=["csv"])
if uploaded_file:
    if st.button("Run Analysis"):
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            
            # Transform the text using the vectorizer
            text_vectorized = vectorizer.transform(uploaded_df['Cleaned_Text'])
            
            # Display feature count info for debugging
            st.info(f"Input features: {text_vectorized.shape[1]}")
            
            # Make predictions
            predictions = classifier.predict(text_vectorized)
            uploaded_df['prediction'] = predictions
            
            st.write("Analysis Results:")
            st.dataframe(uploaded_df)
            st.write("Prediction Value Counts:")
            st.write(uploaded_df['prediction'].value_counts())
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
