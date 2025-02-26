import streamlit as st
import pandas as pd
import re
import plotly.express as px  # For interactive charts

st.title("Intelligent Process Automation (IPA) Detection")

def detect_ipa_detailed(text):
    keywords = ["automation", "process", "data entry", "document processing", "customer service", "workflow", "robotic", "rpa", "ai", "intelligent"]
    if isinstance(text, str):
        text = text.lower()
        matching_keywords = []
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                matching_keywords.append(keyword)

        if matching_keywords:
            confidence = len(matching_keywords) / len(keywords)
            return "IPA Related", ", ".join(matching_keywords), confidence
        else:
            return "Not IPA Related", "", 0
    else:
        return "Not IPA Related", "", 0

uploaded_file = st.file_uploader("Upload CSV file for IPA detection", type=["csv"])

if uploaded_file:
    if st.button("Detect IPA"):
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            ipa_results = []
            for text in uploaded_df['Cleaned_Text']:
                ipa_results.append(detect_ipa_detailed(text))

            uploaded_df[['IPA Detection', 'Matching Keywords', 'Confidence']] = pd.DataFrame(ipa_results, index=uploaded_df.index)

            # --- Visualizations ---
            st.subheader("IPA Detection Visualizations")

            # Bar Chart
            detection_counts = uploaded_df['IPA Detection'].value_counts()
            fig_bar = px.bar(detection_counts, x=detection_counts.index, y=detection_counts.values, labels={'y': 'Count', 'x': 'IPA Detection'})
            st.plotly_chart(fig_bar)

            # Pie Chart
            fig_pie = px.pie(detection_counts, names=detection_counts.index, values=detection_counts.values, title="IPA Detection Distribution")
            st.plotly_chart(fig_pie)

            # --- Detailed Results ---
            st.subheader("Detailed Results")
            for index, row in uploaded_df.iterrows():
                if row['IPA Detection'] == "IPA Related":
                    st.write(f"**Text:** {row['Text']}")
                    st.write(f"**Cleaned Text:** {row['Cleaned_Text']}")
                    st.write(f"**IPA Detection:** {row['IPA Detection']}")
                    st.write(f"**Matching Keywords:** {row['Matching Keywords']}")
                    st.write(f"**Confidence:** {row['Confidence']:.2f}")
                    st.write("---")

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
