import pandas as pd
import streamlit as st
import numpy as np
from app import *
import os
# st.write("Here's our first attempt at using data to create a table:")
# st.write(pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# }))


# dataframe = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))

# st.dataframe(dataframe.style.highlight_max(axis=1))

# dataframe = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))
# st.table(dataframe)

# st.text_input("Your name", key="name")
# x = st.slider('x')  # 👈 this is a widget
# st.write(x, 'squared is', x * x)

# if st.checkbox('Show dataframe'):
#     chart_data = pd.DataFrame(
#        np.random.randn(20, 3),
#        columns=['a', 'b', 'c'])

#     chart_data

# df = pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
#     })

# option = st.selectbox(
#     'Which number do you like best?',
#      df['second column'])

# 'You selected: ', option

# add_selectbox = st.sidebar.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone')
# )

# # Add a slider to the sidebar:
# add_slider = st.sidebar.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0)
# )

# left_column,center_column, right_column  = st.columns(3)
# # You can use a column just like st.sidebar:
# left_column.button('Press me!')

# # Or even better, call Streamlit functions inside a "with" block:
# with right_column:
#     chosen = st.radio(
#         'Sorting hat',
#         ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
#     st.write(f"You are in {chosen} house!")

# import time

# 'Starting a long computation...'

# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.1)

# '...and now we\'re done!'

# st.markdown("# JD - CV scorer 🎈")
# left_column,center_column, right_column  = st.columns(3)

# with left_column:
#     uploaded_file = st. file_uploader('upload you JD here')
#     folder=r'D:/jdcv_score_app/jdcv_score_app/JD/'
#     if uploaded_file is not None:
#         st.write("filename:", uploaded_file)
#         print('bbbbbbbbbbbb',uploaded_file)
#         JD_embedding = jd_embedding(folder+uploaded_file.name)
#         st.write(JD_embedding)

# with right_column:
#     folder_resume=r'D:/jdcv_score_app/jdcv_score_app/resume/'
#     uploaded_files = st.file_uploader("Choose the folder of the resume", accept_multiple_files=True)
#     for i in uploaded_files:
#         if i is not None:
#             st.write("filename:", i)
#             print('bbbbbbbbbbbb',i)
#             RESUME_embedding = RESUME_embedding(folder_resume)
#             st.write(RESUME_embedding)

# with center_column:
#     score_dict = {}
#     for filename, resume_embedding in RESUME_embedding.items():
#         score = 1 - cosine(JD_embedding, resume_embedding)
#         score_dict[filename] = score * 100

#     sorted_dict_desc = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))
#     st.write(sorted_dict_desc)



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

    # with right_column:
    #     folder_resume = r'D:/jdcv_score_app/jdcv_score_app/resume/'
    #     uploaded_resume_files = st.file_uploader(
    #         "Choose the all of the resumes", accept_multiple_files=True
    #     )
    #     resume_embeddings = {}  # Dictionary to store resume embeddings

    #     if uploaded_resume_files:
    #         for uploaded_resume_file in uploaded_resume_files:
    #             if uploaded_resume_file is not None:
    #                 st.write("Resume filename:", uploaded_resume_file.name)
    #                 print('Uploaded resume:', uploaded_resume_file.name)  # Optional output

    #                 try:
    #                     # Assuming you have logic to calculate resume embedding
    #                     # based on the uploaded file, update the following line:
    #                     document = SimpleDirectoryReader(input_files=[folder_resume + uploaded_resume_file.name]).load_data()
    #                     REsume_embedding = embedding_model(preprocessing(document))
    #                     resume_embeddings[uploaded_resume_file.name] = REsume_embedding # Calculate resume embedding using your function
    #                     # resume_embeddings[uploaded_resume_file.name] = resume_embedding  # Store with filename
    #                     st.write("Resume uploaded")
    #                 except Exception as e:
    #                     st.error(f"Error processing resume {uploaded_resume_file.name}: {e}")

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



