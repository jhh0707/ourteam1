import streamlit as st

#전체 레이아웃을 넓게 설정
st.set_page_config(layout="wide")

#제목 설정
st.title("타격폼의 주인공을 찾아라")

#파일 업로드 버튼을 상단으로 이동
uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])

#전체 레이아웃을 컨테이너로 감싸기
with st.container():
    col1, col2 = st.columns(2)  # 열을 균등하게 분배하여 넓게 표시

    #파일 업로드
    #uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])

    with col1:
        st.header("원본 영상")
        if uploaded_file is not None:
            st.video(uploaded_file)
        else:
            st.write("원본 영상을 표시하려면 비디오 파일을 업로드하세요.")

    with col2:
        st.header("인물 검출 결과 영상")
        if "processed_video" in st.session_state:
            st.video(st.session_state["processed_video"])
        else:
            #st.write("여기에 인물 검출 결과가 표시됩니다.")
            result_placeholder.markdown(
                """
                <div style='width:100%; height:620px; background-color:#d3d3d3; display:flex; align-items:center; justify-content:center; border-radius:5px;'>
                    <p style='color:#888;'>여기에 인물 검출 결과가 표시됩니다.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# 사물 검출 버튼 추가
if st.button("인물 검출 실행"):
    if uploaded_file is not None:
        st.session_state["processed_video"] = uploaded_file
        st.success("인물 검출이 완료되어 오른쪽에 표시됩니다.")
    else:
        st.warning("인물 검출을 실행하려면 비디오 파일을 업로드하세요.")
