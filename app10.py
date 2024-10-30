#필요한 모듈 불러오기
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
df


#유저의 질문을 입력받아 가장 유사한 질문의 답변을 반환하는 함수를 작성합니다.
def greet(user, history=[]):
    #사용자 입력 임베딩(숫자로 변환)
    embedding = encoder.encode(user)

    #유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    #히스토리에 추가하여 대화 유지
    history.append([user, answer['챗봇']])
    return history, history

#Gradio의 인터페이스를 통해 greet 함수를 대화형으로 사용할 수 있도록 설정합니다.
demo = gr.Interface(
    fn=greet,
    inputs=["text", "state"],
    outputs=["chatbot", "state"],
    title="타격폼의 주인공 찾기",
    description="프로젝트에 관한 질문을 입력해보세요. 예: 포트폴리오 주제가 뭔가요?"
)

# 챗봇 실행
demo.launch(debug=True, share=True)