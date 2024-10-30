#필요한 모듈 불러오기
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 식당 관련 질문과 답변 데이터
questions = [
    "포트폴리오 주제가 뭔가요?",
    "모델은 어떤걸 썼나요?",
    "프로젝트 구성원은 어떻게 되나요?",
    "프로젝트 기간은 어떻게 되나요?",
    "조장이 누구인가요?",
    "데이터는 뭘 이용했나요?",
    "프로젝트 진행시 어려움은 없었나요?"
]

answers = [
    "야구선수의 타격폼으로 누군지 구분하는 것입니다.",
    "Yolo모델 8버전을 사용했습니다.",
    "한지희, 안민영입니다.",
    "10월 28일 ~ 11월 18일 입니다.",
    "한지희입니다.",
    "야구 중계 데이터를 직접 녹화하여 사용했습니다.",
    "프레임별로 라벨링을 해야해서 전처리 해야할 것이 많아 힘들었습니다."
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("타격폼의 주인공 찾기")
st.write("프로젝트에 관한 질문을 입력해보세요. 예: 포트폴리오 주제가 뭔가요?")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")
