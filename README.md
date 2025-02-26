# Document Classification and Topic Extraction

## Overview
This repository provides a document classification and similarity search system. Given a query, the system retrieves top-matching documents along with their assigned topics and extracted text. It is designed for text-based document understanding and information retrieval.

The model classifies uploaded documents and extracts key topics from the text using NLP techniques. The user can test different queries to find semantically similar documents in the dataset.
## Example Output
Below is an example of the system in action:

![image](https://github.com/user-attachments/assets/b1c5f0ff-7f67-43df-a59f-a58ae3055bed)

## Features
- Document classification using machine learning models
- Topic extraction and analysis
- Interactive visualization of classification results
- Support for CSV file uploads
- Pre-trained models for immediate use

## Repository Contents
* `streamlit_app.py` – The main Streamlit application for document classification.
* `model-training.ipynb` – Jupyter Notebook for training the machine learning model.
* `lda_model1.pkl` – Pre-trained LDA model for topic extraction.
* `vectorizer.pkl` – Pre-trained vectorizer for text processing.
* `setup.sh` – Shell script for setting up the environment.
* `requirements.txt` – List of required Python packages.
* `processed_text_data.csv` – Sample processed dataset.
* `synthetic_data.csv` – Additional synthetic text dataset.
* `test_documents.csv` – Example test dataset for classification.
* `train_extracted_text_cleaned.csv` – Preprocessed training data.
* `FormUnderstanding_data.zip` – The dataset used for training.
* `LICENSE` – License information.

## How to Set Up the Environment
You can run this repository on **Streamlit Cloud** or a local environment.

### Running on Streamlit Cloud
1. Clone this repository to **Streamlit Cloud**.
2. The Streamlit app will automatically run `streamlit_app.py`.
3. The `test_documents.csv` is already available in the repo.
4. Upload a different test file if needed (ensure it follows the correct format).

### Running Locally
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo.git
   cd your-repo
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the setup script (optional):
   ```sh
   bash setup.sh
   ```

5. Start the Streamlit application:
   ```sh
   streamlit run streamlit_app.py
   ```

## How to Use the App
1. Upload a test file via the Streamlit app.
2. The model will classify the document and extract key topics.
3. View the results in an interactive interface.

## Test File Format
The test file should be a **CSV** with the following format:
```
Image,Extracted_Text
doc1.png,This is a financial report for the year 2023.
doc2.png,Research on AI ethics and fairness in machine learning models.
...
```

To change the test file used, modify the file path in `streamlit_app.py`.

## Requirements
- Python 3.8 or higher
- Dependencies listed in `requirements.txt`



## Troubleshooting
- **Model loading issues**: Ensure that all pickle files are in the correct location
- **CSV format errors**: Verify your CSV follows the format specified above
- **Memory errors**: For large files, consider processing in smaller batches

## License
This project is open-source under the LICENSE.

## Contributors
Mahi Mann


