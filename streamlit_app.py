import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Load Model ---
st.title("📄 Document Topic Classification & Similarity Search")

try:
    model_data = joblib.load("lda_model1.pkl")
    lda = model_data["lda"]
    vectorizer = model_data["vectorizer"]
    topic_names = model_data["topic_names"]  # List of predefined topics
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- Text Cleaning Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# --- Load Documents ---
try:
    doc_df = pd.read_csv("test_documents(1).csv")  # Ensure this file exists
    doc_df['cleaned_text'] = doc_df['Extracted_Text'].fillna('').apply(clean_text)
    doc_matrix = vectorizer.transform(doc_df['cleaned_text'])  # Transform documents
except Exception as e:
    st.error(f"❌ Error loading documents: {e}")
    st.stop()

# --- Function to Predict Topic for a Document ---
def predict_topic(text_vector, lda_model, topic_names):
    topic_distribution = lda_model.transform(text_vector)
    top_topic_idx = np.argmax(topic_distribution)
    return topic_names[top_topic_idx]  # Return most relevant topic

# Assign topics to each document
doc_df['Assigned_Topic'] = [
    predict_topic(doc_matrix[i], lda, topic_names) for i in range(len(doc_df))
]

# --- Function to Get Top Matches ---
def get_top_matches(query, vectorizer, lda_model, doc_matrix, doc_df, top_n=5):
    query_cleaned = clean_text(query)
    query_vec = vectorizer.transform([query_cleaned])
    query_topic_dist = lda_model.transform(query_vec)
    doc_topic_dist = lda_model.transform(doc_matrix)

    similarities = cosine_similarity(query_topic_dist, doc_topic_dist).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    return doc_df.iloc[top_indices][['Image', 'Extracted_Text', 'Assigned_Topic']], similarities[top_indices]

# --- Streamlit Interface ---
query = st.text_input("🔍 Enter a query to test:")
if st.button("Search"):
    if not query.strip():
        st.warning("⚠️ Please enter a valid query!")
    else:
        results, scores = get_top_matches(query, vectorizer, lda, doc_matrix, doc_df)
        
        if scores.max() == 0:
            st.warning("⚠️ No relevant documents found.")
        else:
            st.subheader("🔝 Top Matching Documents:")
            for i, (index, row) in enumerate(results.iterrows()):
                st.write(f"**{i+1}. Image:** {row['Image']} **(Score: {scores[i]:.4f})**")
                st.write(f"🏷 **Assigned Topic:** {row['Assigned_Topic']}")  # Show inferred topic
                st.write(f"📄 **Extracted Text:** {row['Extracted_Text'][:300]}...")  # Show preview
                st.write("---")
