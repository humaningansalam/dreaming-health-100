import streamlit as st
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS

@st.cache_resource
def get_video_vectorstore():
    return FAISS.load_local(
        folder_path='./data/video_vectorstore', 
        embeddings=CohereEmbeddings(
            model="embed-multilingual-v3.0",
            cohere_api_key=st.secrets["cohere_api_key"]
        ), 
        allow_dangerous_deserialization=True
    )
