import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Model ---
@st.cache_resource()
def load_model():
    return joblib.load("lda_model1.pkl")

model_data = load_model()
lda_model = model_data["lda"]
vectorizer = model_data["vectorizer"]
topic_names = model_data["topic_names"]

# --- Text Cleaning Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# --- Load Dataset ---
df = pd.read_csv("large_synthetic_text_dataset.csv")
df["cleaned_text"] = df["Extracted_Text"].fillna('').apply(clean_text)
X_corpus = vectorizer.transform(df["cleaned_text"])

# --- Streamlit UI ---
st.title("📄 Document Topic Classification & Similarity Search")
query = st.text_input("Enter a query to test:")

if query:
    query_clean = clean_text(query)
    query_vector = vectorizer.transform([query_clean])
    
    # Compute similarity
    similarity_scores = cosine_similarity(query_vector, X_corpus)[0]
    df["Similarity"] = similarity_scores
    df_sorted = df.sort_values(by="Similarity", ascending=False).head(5)
    
    # Display results
    st.subheader("🔝 Top Matching Documents:")
    for _, row in df_sorted.iterrows():
        st.write(f"1️⃣ **Image:** {row['Image']} (**Score:** {row['Similarity']:.4f})")
        st.write(f"📄 **Extracted Text:** {row['Extracted_Text']}")
        st.write(f"🏷 **Assigned Topic:** {row['Assigned_Topic']}")
        st.write("---")
