import streamlit as st
from langchain_cohere import ChatCohere

@st.cache_resource
def get_chat_model():
    return ChatCohere(
        model="command-r-plus-08-2024",  # 최신 Cohere 모델 선택
        cohere_api_key=st.secrets["cohere_api_key"]  # Streamlit secrets에서 API 키 가져오기
    )