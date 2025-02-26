import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load("lda_model1.pkl")
        lda = model_data["lda"]
        vectorizer = model_data["vectorizer"]
        topic_names = model_data["topic_names"]
        return lda, vectorizer, topic_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

lda, vectorizer, topic_names = load_model()

# --- Text Cleaning Function ---
def clean_text(text):
    if not isinstance(text, str):  # Ensure input is a string
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# --- Function to Predict Document Topic ---
def predict_topic(text, lda, vectorizer, topic_names):
    text_clean = clean_text(text)
    text_vectorized = vectorizer.transform([text_clean])
    topic_distribution = lda.transform(text_vectorized)
    topic_index = topic_distribution.argmax()
    topic_name = topic_names[topic_index]
    return topic_name, topic_distribution[0][topic_index]

# --- Function to Find Similar Documents ---
def get_top_matches(query, vectorizer, doc_matrix, doc_df, top_n=5):
    query_vec = vectorizer.transform([query])  # Vectorize query
    similarities = cosine_similarity(query_vec, doc_matrix).flatten()  # Compute similarity
    top_indices = similarities.argsort()[-top_n:][::-1]  # Get top N indices
    return doc_df.iloc[top_indices][['Image', 'Extracted_Text']], similarities[top_indices]

# --- Streamlit UI ---
st.title("📄 Document Topic Classification & Similarity Search")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload a CSV file with extracted text", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Ensure 'Extracted_Text' column exists
        if 'Extracted_Text' not in df.columns:
            st.error("Error: 'Extracted_Text' column not found in uploaded file.")
            st.stop()

        # Clean text and vectorize
        df['cleaned_text'] = df['Extracted_Text'].fillna('').apply(clean_text)
        doc_matrix = vectorizer.transform(df['cleaned_text'])

        # Predict topics
        df['Predicted_Topic'] = df['cleaned_text'].apply(lambda x: predict_topic(x, lda, vectorizer, topic_names)[0])

        st.success("✅ Topic classification complete!")
        st.write(df[['Image', 'Predicted_Topic']])

        # --- Query-based Similarity Search ---
        query = st.text_area("🔎 Enter a query to find similar documents:")

        if query:
            top_matches, similarity_scores = get_top_matches(query, vectorizer, doc_matrix, df)
            st.subheader("🔝 Top Matching Documents:")
            
            for idx, (index, row) in enumerate(top_matches.iterrows()):
                st.write(f"**{idx+1}. Image:** {row['Image']} (Score: {similarity_scores[idx]:.4f})")
                st.write(f"**Extracted Text:** {row['Extracted_Text'][:300]}...")  # Show first 300 chars
                st.write("---")

    except Exception as e:
        st.error(f"Error processing file: {e}")
