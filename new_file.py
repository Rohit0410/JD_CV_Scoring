import os
import streamlit as st
from tempfile import NamedTemporaryFile
from llama_index.core import SimpleDirectoryReader
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import fitz
from docx import Document
import glob

# Download NLTK resources
nltk.download('stopwords')

# Load the SentenceTransformer model
model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

# Initialize NLTK stopwords
stop_words = set(stopwords.words('english'))

def preprocessing(document):
    """Preprocesses text data."""
    text1 = document[0].text.replace('\n', '').lower()
    text = re.sub(r'[^\x00-\x7F]+', '', text1)  # Remove non-ASCII characters
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]  # Remove punctuation
    tokens = [token for token in tokens if token]  # Remove empty tokens
    filtered_tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

def embedding_model(data):
    """Encodes data using the provided model."""
    text_embedding = model.encode(data)
    return text_embedding

def jd_embedding(uploaded_jd_file):
    """Computes the embedding for a job description (JD) file."""
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_filename = tmp_file.name
        tmp_file.write(uploaded_jd_file.read())
    document = SimpleDirectoryReader(input_files=[tmp_filename]).load_data()
    data = preprocessing(document)
    JD_embedding = embedding_model(data)
    os.unlink(tmp_filename)  # Delete temporary file
    return JD_embedding

def debug_app():
    st.markdown('''# TalentMatch360 🌟''')
                
    st.markdown('''### Nextgen tool for evaluating Job Descriptions and Resumes''')

    left_column, center_column, right_column = st.columns(3)

    with left_column:
        uploaded_jd_file = st.file_uploader("Upload your JD here")
        
        if uploaded_jd_file is not None:
            try:
                JD_embedding = jd_embedding(uploaded_jd_file)
                st.write("JD uploaded")
            except Exception as e:
                st.error(f"Error processing JD: {e}")
        

    
    with right_column:
        uploaded_resume_files = st.file_uploader(
            "Upload all of the resumes", accept_multiple_files=True
        )
        resume_embeddings = {}  # Dictionary to store resume embeddings

        if uploaded_resume_files:
            for uploaded_resume_file in uploaded_resume_files:
                if uploaded_resume_file is not None:
                    try:
                        with NamedTemporaryFile(delete=False) as tmp_file:
                            tmp_filename = tmp_file.name
                            tmp_file.write(uploaded_resume_file.read())
                        document = SimpleDirectoryReader(input_files=[tmp_filename]).load_data()
                        data = preprocessing(document)
                        REsume_embedding = embedding_model(data)
                        resume_embeddings[uploaded_resume_file.name] = REsume_embedding
                        os.unlink(tmp_filename)  # Delete temporary file
                    except Exception as e:
                        st.error(f"Error processing resume {uploaded_resume_file.name}: {e}")
            st.write("Resume uploaded")

    with center_column:
        score_dict = {}
        for filename, resume_embedding in resume_embeddings.items():
            score = 1 - cosine(JD_embedding, resume_embedding)
            score_dict[filename] = score * 100
            
        sorted_dict_desc = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame(list(sorted_dict_desc.items()), columns=['Resume', 'Score'])
        st.dataframe(df, width=1200)

if __name__ == "__main__":
    debug_app()
