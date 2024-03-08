import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
from constant import model
from llama_index.core import SimpleDirectoryReader
import os
from scipy.spatial.distance import cosine
import fitz
from docx import Document

import glob

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

def scoring(folder, file):
    
    """Scores resumes in a folder based on their similarity to a job description (JD).

    Args:
        folder: The path to the folder containing resumes.
        file: The path to the JD file.

    Prints:
        A sorted dictionary of filenames and their similarity scores to the JD (descending order).
    """

    JD_embedding = jd_embedding(file)
    REsume_embedding = RESUME_embedding(folder)
    score_dict = {}

    for filename, resume_embedding in REsume_embedding.items():
        score = 1 - cosine(JD_embedding, resume_embedding)
        score_dict[filename] = score * 100

    sorted_dict_desc = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))
    print('ff',sorted_dict_desc)
    return sorted_dict_desc

# scoring(r'D:/jdcv_score_app/jdcv_score_app/resume/', r'JD\Head of Insurance (OLA) JD.docx')
