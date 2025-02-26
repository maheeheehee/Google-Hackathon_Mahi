import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    doc_topic_dist = lda.transform(doc_matrix)
except Exception as e:
    st.error(f"Error loading documents: {e}")
    st.stop()

# --- Function to Assign Topics ---
def assign_topic(topic_distribution, topic_names):
    max_topic_idx = np.argmax(topic_distribution)  # Get index of most probable topic
    return topic_names[max_topic_idx]

doc_df['Assigned_Topic'] = [assign_topic(topic_dist, topic_names) for topic_dist in doc_topic_dist]

# --- Function to Get Top Matches ---
def get_top_matches(query, vectorizer, lda_model, doc_matrix, doc_df, top_n=5):
    query_cleaned = clean_text(query)
    query_vec = vectorizer.transform([query_cleaned])
    query_topic_dist = lda_model.transform(query_vec)
    
    similarities = cosine_similarity(query_topic_dist, doc_topic_dist).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return doc_df.iloc[top_indices][['Image', 'Extracted_Text', 'Assigned_Topic']], similarities[top_indices]

# --- Streamlit Interface ---
query = st.text_input("Enter a query to test:")
if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid query!")
    else:
        results, scores = get_top_matches(query, vectorizer, lda, doc_matrix, doc_df)
        
        if scores.max() == 0:
            st.warning("No relevant documents found.")
        else:
            st.subheader("🔝 Top Matching Documents:")
            for i, (index, row) in enumerate(results.iterrows()):
                st.write(f"**{i+1}. Image:** {row['Image']} **(Score: {scores[i]:.4f})**")
                st.write(f"🏷 **Assigned Topic:** {row['Assigned_Topic']}")
                st.write(f"📄 **Extracted Text:** {row['Extracted_Text'][:300]}...")  # Show preview
                st.write("---")
