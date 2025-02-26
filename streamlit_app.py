import streamlit as st
import pandas as pd
import re
import plotly.express as px

st.title("Intelligent Process Automation (IPA) Detection")

def detect_ipa_detailed(text):
    # ... (same as before)

uploaded_file = st.file_uploader("Upload CSV file for IPA detection", type=["csv"])

if uploaded_file:
    if st.button("Detect IPA"):
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            ipa_results = []
            for text in uploaded_df['Cleaned_Text']:
                ipa_results.append(detect_ipa_detailed(text))

            uploaded_df[['IPA Detection', 'Matching Keywords', 'Confidence']] = pd.DataFrame(ipa_results, index=uploaded_df.index)

            # --- Summary Metrics ---
            total_texts = len(uploaded_df)
            ipa_related_count = len(uploaded_df[uploaded_df['IPA Detection'] == "IPA Related"])
            ipa_related_percentage = (ipa_related_count / total_texts) * 100

            st.subheader("IPA Detection Summary")
            st.write(f"Total texts analyzed: {total_texts}")
            st.write(f"IPA Related texts: {ipa_related_count} ({ipa_related_percentage:.2f}%)")

            # --- Visualizations ---
            st.subheader("IPA Detection Distribution")
            detection_counts = uploaded_df['IPA Detection'].value_counts()
            fig_bar = px.bar(detection_counts, x=detection_counts.index, y=detection_counts.values, labels={'y': 'Count', 'x': 'IPA Detection'}, color=detection_counts.index) #added color
            st.plotly_chart(fig_bar)

            # --- Examples of IPA Related Texts ---
            st.subheader("Examples of IPA Related Texts")
            ipa_related_df = uploaded_df[uploaded_df['IPA Detection'] == "IPA Related"].head(5)  # Show top 5
            for index, row in ipa_related_df.iterrows():
                st.write(f"**Text:** {row['Text']}")
                st.write(f"**Matching Keywords:** {row['Matching Keywords']}")
                st.write("---")

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
