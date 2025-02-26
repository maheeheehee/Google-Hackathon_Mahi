import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Load Model ---
st.title("Document Topic Classification & Similarity Search")

try:
    model_data = joblib.load("lda_model1.pkl")
    lda = model_data["lda"]
    vectorizer = model_data["vectorizer"]
    topic_names = model_data["topic_names"]
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Text Cleaning Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# --- Load Documents ---
try:
    doc_df = pd.read_csv("test_documents.csv")  # Ensure this file exists
    doc_df['cleaned_text'] = doc_df['Extracted_Text'].fillna('').apply(clean_text)
    doc_matrix = vectorizer.transform(doc_df['cleaned_text'])  # Transform documents
except Exception as e:
    st.error(f"Error loading documents: {e}")
    st.stop()

# --- Function to Get All Matches ---
def get_matches(query, vectorizer, lda_model, doc_matrix, doc_df, topic_names):
    query_cleaned = clean_text(query)
    query_vec = vectorizer.transform([query_cleaned]).toarray()  # Convert sparse to dense
    query_topic_dist = lda_model.transform(query_vec)
    doc_topic_dist = lda_model.transform(doc_matrix.toarray())  # Convert sparse to dense

    similarities = cosine_similarity(query_topic_dist, doc_topic_dist).flatten()
    sorted_indices = np.argsort(similarities)[::-1]  # Sort by highest similarity

    matched_docs = doc_df.iloc[sorted_indices].copy()
    matched_docs["Assigned_Topic"] = [
        topic_names[np.argmax(lda_model.transform(vectorizer.transform([text]).toarray())[0])]
        for text in matched_docs["cleaned_text"]
    ]

    return matched_docs[['Image', 'Extracted_Text', 'Assigned_Topic']], similarities[sorted_indices]

# --- Streamlit Interface ---
query = st.text_input("Enter a query to test:")
if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid query!")
    else:
        results, scores = get_matches(query, vectorizer, lda, doc_matrix, doc_df, topic_names)

        if scores.max() == 0:
            st.warning("No relevant documents found.")
        else:
            st.subheader(f"🔝 Top {len(results)} Matching Documents:")
            for i, (index, row) in enumerate(results.iterrows()):
                st.write(f"**{i+1}. Image:** {row['Image']} **(Score: {scores[i]:.4f})**")
                st.write(f"🏷 **Assigned Topic:** {row['Assigned_Topic']}")
                st.write(f"📄 **Extracted Text:** {row['Extracted_Text'][:300]}...")  # Show preview
                st.write("---")
