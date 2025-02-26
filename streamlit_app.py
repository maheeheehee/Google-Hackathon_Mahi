import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import streamlit as st
import sklearn

print(f"Using scikit-learn version: {sklearn.__version__}")

# Load dataset
file_path = "matched_form_data.csv" #Make sure the csv is in same directory as this file.
df = pd.read_csv(file_path)
print(f"Data loaded with {df.shape[0]} rows")

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Cleaned_Text'])
y = df['Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Model training complete")

# Save both the classifier and vectorizer together
model_data = {
    'classifier': clf,
    'vectorizer': vectorizer
}
joblib.dump(model_data, "my_model_new1.pkl")
print("Model and vectorizer saved successfully as 'my_model_new1.pkl'")

st.title("Model Analysis App")

# Load the model and vectorizer together
try:
    model_data = joblib.load("my_model_new1.pkl")
    classifier = model_data['classifier']
    vectorizer = model_data['vectorizer']
    st.success("Model and vectorizer loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload CSV file for analysis", type=["csv"])
if uploaded_file:
    if st.button("Run Analysis"):
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            text_vectorized = vectorizer.transform(uploaded_df['Cleaned_Text'])
            predictions = classifier.predict(text_vectorized)
            uploaded_df['prediction'] = predictions

            st.write("Analysis Results:")
            st.dataframe(uploaded_df)

            st.write("Prediction Value Counts:")
            st.write(uploaded_df['prediction'].value_counts())

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
