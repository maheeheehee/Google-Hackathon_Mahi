import streamlit as st
import pandas as pd

st.title("Intelligent Process Automation (IPA) Detection")

def detect_ipa(text):
    """Detects if text relates to Intelligent Process Automation (IPA)."""
    keywords = ["automation", "process", "data entry", "document processing", "customer service", "workflow", "robotic", "rpa", "ai", "intelligent"]
    if isinstance(text, str): #added to check if text is a string
        text = text.lower()
        for keyword in keywords:
            if keyword in text:
                return "IPA Related"
        return "Not IPA Related"
    else:
        return "Not IPA Related"

uploaded_file = st.file_uploader("Upload CSV file for IPA detection", type=["csv"])

if uploaded_file:
    if st.button("Detect IPA"):
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            ipa_labels = []
            for text in uploaded_df['Cleaned_Text']:
                ipa_labels.append(detect_ipa(text))
            uploaded_df['IPA Detection'] = ipa_labels

            st.write("IPA Detection Results:")
            st.dataframe(uploaded_df)

            st.write("IPA Detection Value Counts:")
            st.write(uploaded_df['IPA Detection'].value_counts())

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
