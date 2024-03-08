import pandas as pd
import streamlit as st
import numpy as np
# from app import *
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# from constant import model
from llama_index.core import SimpleDirectoryReader
import os
from scipy.spatial.distance import cosine
import fitz
from docx import Document
import glob

from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
# model = SentenceTransformer("avsolatorio/GIST-Embedding-v0")
model=SentenceTransformer("NeuML/pubmedbert-base-embeddings")

def get_file_path_by_name(file_name):
    file_path = None
    for path in glob.glob(f'**/{file_name}', recursive=True):
        if file_name in path:
            file_path = path
            break
        file_path = os.path.abspath(file_path)
    return file_path


def preprocessing(document):
    """Preprocesses text data.

    Args:
        document: A list containing the text data.

    Returns:
        The preprocessed text as a string.
    """

    text1 = document[0].text.replace('\n', '').lower()
    text = re.sub(r'[^\x00-\x7F]+', '', text1)  # Remove non-ASCII characters
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]  # Remove punctuation
    tokens = [token for token in tokens if token]  # Remove empty tokens
    filtered_tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

def embedding_model(data):
    """Encodes data using the provided model.

    Args:
        data: The text data to encode.

    Returns:
        The encoded representation of the data.
    """

    text_embedding = model.encode(data)
    return text_embedding

def jd_embedding(uploaded_jd_file):
    """Computes the embedding for a job description (JD) file.

    Args:
        file: The path to the JD file.

    Returns:
        The embedding of the JD.
    """

    document = SimpleDirectoryReader(input_files=[uploaded_jd_file]).load_data()
    print('yes its started')
    # document = file_reader(uploaded_jd_file)
    print('yes')
    data1 = preprocessing(document)
    print('yes')
    JD_embedding = embedding_model(data1)
    return JD_embedding

# def RESUME_embedding(folder):
#     """Computes the embedding for each resume in a folder.

#     Args:
#         folder: The path to the folder containing resumes.

#     Returns:
#         A dictionary mapping filenames to their corresponding embeddings.
#     """

#     resume_embeddings = {}
#     for filename in os.listdir(folder):
#         if os.path.isfile(os.path.join(folder, filename)):
#             print(filename)
#             # document = SimpleDirectoryReader(input_files=[folder + filename]).load_data()
#             document = file_reader(filename)
#             REsume_embedding = embedding_model(preprocessing(document))
#             resume_embeddings[filename] = REsume_embedding
#     return resume_embeddings



def debug_app():
    st.markdown("# JD - CV scorer ")
    left_column,right_column = st.columns(2)

    with left_column:
        uploaded_jd_file = st.file_uploader("Upload your JD here")
        # folder_jd = r'D:/jdcv_score_app/jdcv_score_app/JD/'

        if uploaded_jd_file is not None:
            try:
                # JD_embedding = jd_embedding(folder_jd+uploaded_jd_file.name)
                filepath=get_file_path_by_name(uploaded_jd_file.name)
                JD_embedding=jd_embedding(filepath)

                st.write("JD uploaded")
            except Exception as e:
                st.error(f"Error processing JD: {e}")
        
        # folder_resume = r'D:/jdcv_score_app/jdcv_score_app/resume/'
        uploaded_resume_files = st.file_uploader(
            "Choose the all of the resumes", accept_multiple_files=True
        )
        resume_embeddings = {}  # Dictionary to store resume embeddings

        if uploaded_resume_files:
            for uploaded_resume_file in uploaded_resume_files:
                if uploaded_resume_file is not None:
                    # st.write("Resume filename:", uploaded_resume_file.name)
                    print('Uploaded resume:', uploaded_resume_file.name)  # Optional output

                    try:
                        # Assuming you have logic to calculate resume embedding
                        # based on the uploaded file, update the following line:
                        filepath=get_file_path_by_name(uploaded_resume_file.name)
                        document = SimpleDirectoryReader(input_files=[filepath]).load_data()
                        REsume_embedding = embedding_model(preprocessing(document))
                        resume_embeddings[uploaded_resume_file.name] = REsume_embedding # Calculate resume embedding using your function
                        # resume_embeddings[uploaded_resume_file.name] = resume_embedding  # Store with filename
                    except Exception as e:
                        st.error(f"Error processing resume {uploaded_resume_file.name}: {e}")
            st.write("Resume uploaded")


    with right_column:
        score_dict = {}

        for filename, resume_embedding in resume_embeddings.items():
            score = 1 - cosine(JD_embedding, resume_embedding)
            score_dict[filename] = score * 100
            
        sorted_dict_desc = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame(list(sorted_dict_desc.items()), columns=['Resume', 'Score'])
        st.dataframe(df, use_container_width=True,hide_index=True)
        # st.write(df)


if __name__ == "__main__":
    debug_app()



