import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.title("📄 Document Topic Classification & Similarity Search")

# --- Load Model ---
try:
    model_data = joblib.load("lda_model1.pkl")
    lda = model_data["lda"]
    vectorizer = model_data["vectorizer"]
    topic_names = model_data["topic_names"]
    topic_words = model_data["topic_words"]
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Text Cleaning Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload a CSV file with Extracted_Text", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if "Extracted_Text" not in df.columns:
        st.error("❌ The file must have an 'Extracted_Text' column!")
        st.stop()

    # --- Preprocess Text ---
    df['cleaned_text'] = df['Extracted_Text'].fillna('').apply(clean_text)
    
    # --- Vectorize and Predict Topic ---
    X_test = vectorizer.transform(df['cleaned_text'])
    topic_distributions = lda.transform(X_test)
    df['Topic'] = topic_distributions.argmax(axis=1)  # Assign most probable topic
    df['Topic Name'] = df['Topic'].apply(lambda x: topic_names[x])

    # --- Display Results ---
    st.subheader("📝 Topic Classification Results")
    st.write(df[['Image', 'Topic Name']])

    # --- Search Function ---
    query = st.text_input("🔍 Enter a search query to find similar documents:")
    
    if query:
        query_vec = vectorizer.transform([clean_text(query)])  # Vectorize query
        similarities = cosine_similarity(query_vec, X_test).flatten()  # Compute similarity
        top_indices = similarities.argsort()[-5:][::-1]  # Get top 5

        st.subheader("🔝 Top Matching Documents")
        for i, idx in enumerate(top_indices):
            st.write(f"**{i+1}. Image: {df.iloc[idx]['Image']}** (Score: {similarities[idx]:.4f})")
            st.write(f"📌 **Extracted Snippet:** {df.iloc[idx]['Extracted_Text'][:300]}...")  # Show snippet
            st.write("---")
