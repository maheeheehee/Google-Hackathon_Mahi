import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    """Cleans extracted text by removing special characters and extra spaces."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# --- Load Model ---
st.title("Document Topic Classification & Similarity Search")

try:
    model_data = joblib.load("lda_model.pkl")
    lda = model_data["lda"]
    vectorizer = model_data["vectorizer"]
    topic_names = model_data.get("topic_names", [f"Topic {i}" for i in range(lda.n_components_)])
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
if uploaded_file:
    file_text = uploaded_file.read().decode("utf-8")
    cleaned_text = clean_text(file_text)
    
    # Vectorize & Predict Topic
    query_vec = vectorizer.transform([cleaned_text])
    topic_probabilities = lda.transform(query_vec)
    predicted_topic = topic_names[topic_probabilities.argmax()]
    
    st.subheader(f"📌 The document is classified under: **{predicted_topic}**")
    
    # --- Retrieve Similar Documents ---
    try:
        doc_df = pd.read_csv("document_database.csv")  # Pre-stored dataset with topics
        doc_df["cleaned_text"] = doc_df["Extracted_Text"].fillna('').apply(clean_text)
        doc_matrix = vectorizer.transform(doc_df["cleaned_text"])
        
        similarities = cosine_similarity(query_vec, doc_matrix).flatten()
        top_indices = similarities.argsort()[-5:][::-1]
        
        st.subheader("🔍 Top Matching Documents:")
        for i, index in enumerate(top_indices):
            doc = doc_df.iloc[index]
            score = similarities[index]
            snippet = " ".join(doc["Extracted_Text"].split()[:40])  # Show first 40 words
            
            st.markdown(f"**📄 {i+1}. {doc['Image']} (Score: {score:.4f})**")
            st.write(f"*Snippet:* {snippet}...")
            st.write("---")
    
    except Exception as e:
        st.error(f"Error retrieving similar documents: {e}")
