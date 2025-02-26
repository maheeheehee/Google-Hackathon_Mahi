import streamlit as st
import joblib
import pandas as pd
import sklearn

st.title("Model Analysis App")
st.write(f"Using scikit-learn version: {sklearn.__version__}")

# Load the model and vectorizer together
try:
    model_data = joblib.load("my_model_new.pkl")  # Use the new model file
    
    # Check if model_data is a dict with both classifier and vectorizer
    if isinstance(model_data, dict) and 'classifier' in model_data and 'vectorizer' in model_data:
        classifier = model_data['classifier']
        vectorizer = model_data['vectorizer']
        st.success("Model and vectorizer loaded successfully")
    else:
        st.error("Model file doesn't contain both classifier and vectorizer")
        st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload CSV file for analysis", type=["csv"])
if uploaded_file:
    if st.button("Run Analysis"):
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            
            # Ensure the Cleaned_Text column exists
            if 'Cleaned_Text' not in uploaded_df.columns:
                st.error("The uploaded CSV must contain a 'Cleaned_Text' column")
                st.stop()
                
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
            
            # Option to download results
            csv = uploaded_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="analysis_results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
            st.exception(e)  # This will display the full traceback for debugging
