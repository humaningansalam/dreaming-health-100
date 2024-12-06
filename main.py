import streamlit as st
import psycopg2
import pandas as pd
from langchain_cohere import ChatCohere
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS

# Streamlit UI 설정
st.title('운동 처방 추천 시스템')
st.subheader('나이, 키, 몸무게를 입력하세요.')

# 사용자 입력 받기
age = st.number_input('나이', min_value=10, max_value=100, step=1)
height = st.number_input('키 (cm)', min_value=100, max_value=250, step=1)
weight = st.number_input('몸무게 (kg)', min_value=30, max_value=200, step=1)

# ChatCohere 인스턴스 캐싱
@st.cache_resource
def get_chat_model():
    return ChatCohere(
        model="command-r-plus-08-2024",  # 최신 Cohere 모델 선택
        cohere_api_key=st.secrets["cohere_api_key"]  # Streamlit secrets에서 API 키 가져오기
    )

@st.cache_resource
def get_video_vectorstore():
    return FAISS.load_local(
        folder_path='video_vectorstore', 
        embeddings=CohereEmbeddings(
            model="embed-multilingual-v3.0",
            cohere_api_key=st.secrets["cohere_api_key"]
        ), 
        allow_dangerous_deserialization=True
    )

# 세션 상태 초기화
if 'page' not in st.session_state:
    st.session_state.page = 0  # 현재 페이지 번호
if 'videos' not in st.session_state:
    st.session_state.videos = []  # 추천 동영상 목록
if 'messages' not in st.session_state:
    st.session_state.messages = []  # 채팅 메시지 저장
if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False  # 대화 시작 여부
if 'prescription' not in st.session_state:
    st.session_state.prescription = ""  # 운동 처방 결과
if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []  # 추가 질문과 응답 저장

# 운동 처방 추천받기 버튼 클릭 시
if st.button('운동 처방 추천받기'):
    try:
        # PostgreSQL 연결 설정
        conn = psycopg2.connect(
            dbname=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PWD"],
            host=st.secrets["DB_URL"],
            port=st.secrets["DB_PORT"]
        )
        cursor = conn.cursor()
        
        # 유사한 데이터를 검색하는 쿼리 실행
        query = """
        SELECT mvm_prscrptn_cn
        FROM users_measurements
        ORDER BY ABS(mesure_age_co - %s) + ABS(CAST(mesure_iem_001_value AS DECIMAL) - %s) 
                 + ABS(CAST(mesure_iem_002_value AS DECIMAL) - %s)
        LIMIT 5 
        """
        cursor.execute(query, (age, height, weight))
        similar_prescriptions = cursor.fetchall()
        print(similar_prescriptions)
        
        # LangChain을 사용해 Cohere 모델 호출
        if similar_prescriptions:
            chat_model = get_chat_model()
            
            # 시스템 메시지와 사용자 메시지 준비
            prescription_texts = [item[0] for item in similar_prescriptions]
            
            # 세션 상태에 messages 저장 (대화 시작)
            st.session_state.messages = [
                SystemMessage(content="""
                당신은 전문 운동 처방사입니다. 
                사용자의 신체 정보와 기존 처방을 바탕으로 구체적이고 실용적인 운동 계획을 제시해주세요.
                """),
                HumanMessage(content=f"""
                사용자 정보:
                - 나이: {age}세
                - 키: {height}cm
                - 몸무게: {weight}kg
                - 고려할 건강 조건: x 

                기존 운동 처방 참고 사항:
                {', '.join(prescription_texts)}

                위 정보를 바탕으로 다음 내용을 포함한 맞춤형 운동 처방을 작성해주세요:
                1. 권장되는 운동 종류
                2. 운동 강도와 시간
                3. 주의사항
                이 외에는 필요없습니다.
                """)
            ]

            # Cohere 모델을 사용하여 운동 처방 생성
            with st.spinner('운동 처방을 생성하고 있습니다...'):
                response = chat_model.invoke(st.session_state.messages)
            
            # 운동 처방 결과를 세션 상태에 저장
            st.session_state.prescription = response.content

            # 대화가 시작되었음을 표시
            st.session_state.conversation_started = True

            # 벡터스토어 가져오기 (초기화 시에만 생성)
            vectorstore = get_video_vectorstore()

            # 검색 기능 정의
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5}  # 상위 5개 결과 반환
            )

            # RetrievalQA 체인 생성
            chain = RetrievalQA.from_llm(
                llm=chat_model,
                retriever=retriever,
                return_source_documents=True
            )

            # 예제 실행
            user_input = f"나이 {age}, 키 {height}, 몸무게 {weight}인 사용자를 위한 운동"
            response = chain.invoke({"query": user_input})

            # 추천 동영상 정보를 추출하여 세션 상태에 저장
            videos = []
            for doc in response["source_documents"]:
                video_info = {}
                lines = doc.page_content.strip().split('\n')
                for line in lines:
                    if line.startswith('제목: '):
                        video_info['title'] = line.replace('제목: ', '')
                    elif line.startswith('동영상링크: '):
                        video_info['url'] = line.replace('동영상링크: ', '')
                    elif line.startswith('이미지링크: '):
                        video_info['thumbnail'] = line.replace('이미지링크: ', '')
                if video_info:
                    videos.append(video_info)
            st.session_state.videos.extend(videos)  # 세션 상태에 저장

            st.session_state.page = 0  # 페이지 초기화

        else:
            st.warning("추천할 운동 처방이 없습니다.")
            
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        st.info("데이터베이스 연결 또는 AI 모델 연결을 확인해주세요.")


# 운동 처방 결과가 있으면 항상 상단에 표시
if st.session_state.prescription:
    st.subheader("🏃‍♂️ 맞춤형 운동 처방")
    st.write(st.session_state.prescription)

# ====== 추가 질문 섹션 ======
if st.session_state.conversation_started:
    st.subheader("❓ 추가 질문이 있으신가요?")
    follow_up = st.text_input("운동 처방에 대해 더 자세히 알고 싶은 점을 질문해주세요", key="follow_up_input")
    
    if st.button("질문 전송", key="follow_up_button"):
        if follow_up:
            chat_model = get_chat_model()
            # 사용자 메시지 추가
            st.session_state.messages.append(HumanMessage(content=follow_up))
            # AI 응답 생성
            with st.spinner('답변을 생성하고 있습니다...'):
                follow_up_response = chat_model.invoke(st.session_state.messages)
            # AI 응답 메시지 추가
            st.session_state.messages.append(follow_up_response)
            # 추가 질문과 응답을 저장
            st.session_state.qa_pairs.append((follow_up, follow_up_response.content))
            st.write(follow_up_response.content)
            
            # 추가 동영상 추천받기
            vectorstore = get_video_vectorstore()
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5}  # 상위 10개 결과 반환
            )
            chain = RetrievalQA.from_llm(
                llm=chat_model,
                retriever=retriever,
                return_source_documents=True
            )
            response = chain.invoke({"query": follow_up})
            
            # 내부에서만 참고한 문서 출력
            print("\n추가 추천 문서:")
            for i, doc in enumerate(response["source_documents"], 1):
                print(f"\n문서 {i}:")
                print(doc.page_content)
            
            # 추가 동영상 정보를 세션 상태에 추가
            new_videos = []
            for doc in response["source_documents"]:
                video_info = {}
                lines = doc.page_content.strip().split('\n')
                for line in lines:
                    if line.startswith('제목: '):
                        video_info['title'] = line.replace('제목: ', '')
                    elif line.startswith('동영상링크: '):
                        video_info['url'] = line.replace('동영상링크: ', '')
                    elif line.startswith('이미지링크: '):
                        video_info['thumbnail'] = line.replace('이미지링크: ', '')
                if video_info and video_info not in st.session_state.videos:
                    new_videos.append(video_info)
            st.session_state.videos.extend(new_videos)
            st.session_state.page = 0  # 페이지 초기화

# 추가 질문과 응답 표시
if st.session_state.qa_pairs:
    st.subheader("💬 대화 내용")
    for question, answer in st.session_state.qa_pairs:
        st.markdown(f"**사용자:** {question}")
        st.markdown(f"**AI:** {answer}")

# ====== 추천 동영상 표시 ======
if st.session_state.videos:
    st.subheader("🎥 추천 운동 비디오 목록")

    def display_videos(videos, page, videos_per_page=3):
        total_pages = (len(videos) - 1) // videos_per_page + 1
        start_idx = page * videos_per_page
        end_idx = start_idx + videos_per_page
        current_videos = videos[start_idx:end_idx]
        
        # 동영상 표시 (한 줄에 3개씩)
        cols = st.columns(videos_per_page)
        for idx, video in enumerate(current_videos):
            col = cols[idx % videos_per_page]
            with col:
                st.image(video['thumbnail'], use_container_width=True)
                st.markdown(f"**[{video['title']}]({video['url']})**")
        
        # 페이지 네비게이션
        prev_col, _, page_info_col, _, next_col = st.columns([1, 2, 2, 2, 1])
        with prev_col:
            if st.button("← 이전", key=f"prev_{page}_{len(videos)}", disabled=(page == 0)):
                st.session_state.page -= 1
        with page_info_col:
            st.markdown(f"**페이지 {page + 1} / {total_pages}**")
        with next_col:
            if st.button("다음 →", key=f"next_{page}_{len(videos)}", disabled=(page >= total_pages - 1)):
                st.session_state.page += 1

    display_videos(st.session_state.videos, st.session_state.page)
