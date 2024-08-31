import streamlit as st
import streamlit_authenticator as stauth
import yaml

from dotenv import load_dotenv
from llm import get_ai_response

# 'user_database.yaml' 파일을 열어서 사용자 데이터베이스를 로드 (사용자 ID 및 PW 등)
with open('user_database.yaml') as file:
    user_db = yaml.load(file, Loader = stauth.SafeLoader)

# Streamlit 애플리케이션의 페이지 설정 (제목과 아이콘 지정)
st.set_page_config(page_title="ROKEY 챗봇", page_icon="./icon/character.png")

col1, col2 = st.columns([1, 8])

with col1:
    st.image("./icon/character.png", width=50)

with col2:
    st.title("두산 로키 챗봇")

# st.title("🤖 두산 로키 챗봇")
st.caption("\"ROKEY BOOTCAMP\"에 관련된 사항을 답해드립니다!")

authenticator = stauth.Authenticate(
    user_db['credentials'],
    user_db['cookie']['name'],
    user_db['cookie']['key'],
    user_db['cookie']['expiry_days'],
    user_db['preauthorized']
)

## 로그인 위젯 렌더링
## log(in/out)(로그인 위젯 문구, 버튼 위치)
## 버튼 위치 = "main" or "sidebar"
name, authentication_status, username = authenticator.login(key="Login",location="main")

# authentication_status : 인증 상태 (실패=>False, 값없음=>None, 성공=>True)
if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    authenticator.logout("Logout","sidebar")
    st.sidebar.title(f"Welcome {name}")

    load_dotenv()

    if 'message_list' not in st.session_state:
        st.session_state.message_list = []

    for message in st.session_state.message_list:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_question := st.chat_input(placeholder="궁금한 내용들을 말씀해주세요!"):
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.message_list.append({"role": "user", "content": user_question})

        with st.spinner("답변을 생성하는 중입니다"):
            ai_response = get_ai_response(user_question)
            with st.chat_message("ai"):
                ai_message = st.write_stream(ai_response)
                st.session_state.message_list.append({"role": "ai", "content": ai_message})