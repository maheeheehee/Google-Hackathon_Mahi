import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Model ---
def load_model():
    try:
        model_data = joblib.load("lda_model1.pkl")
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- Text Cleaning Function ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# --- Get Top Matches ---
def get_top_matches(query, vectorizer, doc_matrix, doc_df, top_n=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return doc_df.iloc[top_indices][['Image', 'Extracted_Text']], similarities[top_indices]

# --- Load the Model ---
st.title("Document Topic Classification & Similarity Search")
model_data = load_model()
lda_model = model_data["lda"]
vectorizer = model_data["vectorizer"]
topic_names = model_data["topic_names"]

df = pd.read_csv("test_documents.csv")  # Ensure this exists

# --- Clean and Vectorize Text ---
df['cleaned_text'] = df['Extracted_Text'].fillna('').apply(clean_text)
doc_matrix = vectorizer.transform(df['cleaned_text'])

# --- User Input Query ---
query = st.text_input("Enter a query to test:")

if query:
    query = clean_text(query)
    results, scores = get_top_matches(query, vectorizer, doc_matrix, df, top_n=5)
    
    if scores[0] == 0:
        st.warning("No relevant documents found.")
    else:
        st.subheader("🔝 Top Matching Documents:")
        for i, (index, row) in enumerate(results.iterrows()):
            st.write(f"**{i+1}. Image:** {row['Image']} (Score: {scores[i]:.4f})")
            st.write(f"📄 **Extracted Text:** {row['Extracted_Text'][:300]}...")  # Show snippet
            st.write("---")
