import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Models ---
@st.cache_resource  # Cache to avoid reloading on every interaction
def load_models():
    try:
        with open("lda_model.pkl", "rb") as f:
            lda_model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return lda_model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please upload `lda_model.pkl` and `vectorizer.pkl` to the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

lda_model, vectorizer = load_models()

# --- Load Dataset for Similarity Search ---
@st.cache_data  # Cache dataset
def load_dataset():
    try:
        df = pd.read_csv("train_extracted_text_cleaned.csv")
        df['cleaned_text'] = df['Extracted_Text'].fillna('')
        X_train = vectorizer.transform(df['cleaned_text'])  # Transform dataset for similarity search
        return df, X_train
    except FileNotFoundError:
        st.error("Dataset file `train_extracted_text_cleaned.csv` not found.")
        st.stop()

train_df, X_train = load_dataset()

# --- Function to Clean Text ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# --- Function to Get Top Matches ---
def get_top_matches(query, vectorizer, doc_matrix, doc_df, top_n=5):
    query_vec = vectorizer.transform([query])  # Vectorize query
    similarities = cosine_similarity(query_vec, doc_matrix).flatten()  # Compute similarity
    top_indices = similarities.argsort()[-top_n:][::-1]  # Get top N indices
    return doc_df.iloc[top_indices][['Image', 'Extracted_Text']], similarities[top_indices]

# --- Streamlit UI ---
st.title("📄 Document Topic Prediction & Similarity Search")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    query_text = uploaded_file.read().decode("utf-8").strip()
    query_cleaned = clean_text(query_text)

    # 1️⃣ **Predict Topic** using LDA
    query_vectorized = vectorizer.transform([query_cleaned])
    topic_distribution = lda_model.transform(query_vectorized)
    predicted_topic = topic_distribution.argmax()

    st.subheader("Predicted Topic:")
    st.write(f"📝 The document is classified under **Topic {predicted_topic}**")

    # 2️⃣ **Find Similar Documents**
    top_docs, scores = get_top_matches(query_cleaned, vectorizer, X_train, train_df)

    st.subheader("Top Matching Documents:")
    for idx, (row, score) in enumerate(zip(top_docs.iterrows(), scores)):
        st.write(f"**{idx+1}. Image:** {row[1]['Image']} (Score: {score:.4f})")
        st.write(f"**Extracted Text:** {row[1]['Extracted_Text'][:500]}...")  # Show part of text
        st.write("---")
