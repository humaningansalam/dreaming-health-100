import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

from src.additional_measurements import additional_measurements  
from src.query_manage import fetch_similar_prescriptions 
from llm.model import get_chat_model
from llm.vectorstore import get_video_vectorstore

user_measurements = {key: None for key in additional_measurements.keys()}
# 기본 페이지 세팅
st.set_page_config(page_title="운동 처방 추천 시스템", layout="wide")

# 캐시된 리소스 초기화
chat_model = get_chat_model()
video_vectorstore = get_video_vectorstore()


# 세션 상태 초기화
if "messages" not in st.session_state:
    # 초기 메시지: 시스템 역할 정의
    st.session_state["messages"] = [
        SystemMessage(content="당신은 전문 운동 처방사입니다.")
    ]
if "prescription" not in st.session_state:
    st.session_state["prescription"] = ""
if "conversation_started" not in st.session_state:
    st.session_state["conversation_started"] = False
if "qa_pairs" not in st.session_state:
    st.session_state["qa_pairs"] = []
if "videos" not in st.session_state:
    st.session_state["videos"] = []
if "page" not in st.session_state:
    st.session_state["page"] = 0

# 사이드바 설정
with st.sidebar:
    st.header("사용자 정보 입력")
    age = st.number_input('나이', min_value=10, max_value=100, step=1)
    height = st.number_input('키 (cm)', min_value=100, max_value=250, step=1)
    weight = st.number_input('몸무게 (kg)', min_value=30, max_value=200, step=1)
    disease = st.multiselect("질환", ["고혈압", "당뇨", "관절염"])

    # 추가 세부항목 입력
    with st.expander("추가 세부항목 입력 (선택 사항)"):

        st.markdown("""
        <a href="https://nfa.kspo.or.kr/reserve/0/selectMeasureItemListByAgeSe.kspo" 
        target="_blank" style="text-decoration:none;">
            ℹ️ 측정 기준 보기
        </a>
        """, unsafe_allow_html=True)       
        
        for key, label in additional_measurements.items():
            user_measurements[key] = st.text_input(label)
    
    generate_button = st.button("처방 생성")

def display_videos(videos, page, videos_per_page=3):
    total_pages = (len(videos) - 1) // videos_per_page + 1
    start_idx = page * videos_per_page
    end_idx = start_idx + videos_per_page
    current_videos = videos[start_idx:end_idx]

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

# 처방 생성 버튼 클릭 시 처방 수행
if generate_button:
    try:
        # 입력된 값만 필터링
        filled_measurements = {key: value for key, value in user_measurements.items() if value}

        # 유사한 처방 데이터 가져오기
        similar_prescriptions = fetch_similar_prescriptions(age, height, weight, filled_measurements)

        #print(similar_prescriptions)

        if similar_prescriptions:
            chat_model = get_chat_model()
            prescription_texts = [item[0] for item in similar_prescriptions]

            #formatted_prescription_texts = format_multiple_prescriptions(prescription_texts)

            disease_str = ", ".join(disease) if disease else "특이 질환 없음"

            user_message_content = f"""
            **사용자 정보:**
            - **나이**: {age}세
            - **키**: {height}cm
            - **몸무게**: {weight}kg
            - **질환**: {disease_str}

            **기존 운동 처방 참고 사항:**
            {', '.join(prescription_texts)}

            **위 정보를 바탕으로 다음 내용의 맞춤형 운동 처방을 작성해주세요:**
            1. 권장되는 운동 종류
            2. 운동 강도와 시간
            3. 주의사항
            """

            # 기존 메시지(시스템 메시지)에 사용자 메시지 추가
            st.session_state.messages.append(HumanMessage(content=user_message_content))
            
            with st.spinner("운동 처방을 생성하고 있습니다..."):
                response = chat_model.invoke(st.session_state.messages)
            
            # 처방 결과 저장
            st.session_state.prescription = response.content
            st.session_state.conversation_started = True
            
            # 처방 결과를 어시스턴트 메시지로 출력
            st.session_state.messages.append(AIMessage(content=st.session_state.prescription))

            # 비디오 벡터스토어
            vectorstore = get_video_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            chain = RetrievalQA.from_llm(
                llm=chat_model,
                retriever=retriever,
                return_source_documents=True
            )

            user_input = f"나이 {age}, 키 {height}, 몸무게 {weight}인 사용자를 위한 운동"
            response = chain.invoke({"query": user_input})

            # 추천 동영상 정보 추출
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
            st.session_state.videos.extend(videos)

        else:
            st.warning("추천할 운동 처방이 없습니다.")
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        st.info("데이터베이스 연결 또는 AI 모델 연결을 확인해주세요.")


# 채팅 UI
st.title("💬 운동 처방 채팅")
st.caption("왼쪽 사이드바에 정보를 입력하고 '처방 생성'을 누르면 결과가 표시됩니다. 이후 추가 질문을 자유롭게 입력해보세요.")

# 기존 대화 내용 표시
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user" if isinstance(msg, HumanMessage) else "system"
    if role == "system":
        # 시스템 메시지는 사용자에게 굳이 보여줄 필요 없음
        continue
    st.chat_message(role).write(msg.content)

# 추가 질문 입력
if st.session_state.conversation_started:
    follow_up = st.chat_input("추가 질문을 입력하세요...")

    if follow_up:
        # 사용자 메시지 추가
        st.session_state.messages.append(HumanMessage(content=follow_up))
        st.chat_message("user").write(follow_up)

        chat_model = get_chat_model()
        with st.spinner("답변을 생성하고 있습니다..."):
            follow_up_response = chat_model.invoke(st.session_state.messages)

        st.session_state.messages.append(follow_up_response)
        st.chat_message("assistant").write(follow_up_response.content)
        st.session_state.qa_pairs.append((follow_up, follow_up_response.content))

        # 추가 동영상 추천
        vectorstore = get_video_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        chain = RetrievalQA.from_llm(
            llm=chat_model,
            retriever=retriever,
            return_source_documents=True
        )
        response = chain.invoke({"query": follow_up})

        # 추가 동영상 정보 추출
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
        st.session_state.page = 0

# 추천 동영상 표시
if st.session_state.videos:
    st.subheader("🎥 추천 운동 비디오 목록")
    display_videos(st.session_state.videos, st.session_state.page)
