from langchain_core.output_parsers import StrOutputParser  # LLM의 출력을 문자열로 변환해주는 파서
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate  # 채팅 프롬프트 및 메시지 템플릿 관련 클래스들
from langchain.chains import create_history_aware_retriever, create_retrieval_chain  # 사용자의 질문과 관련된 과거 대화 기록을 고려하는 검색 체인 생성 도구
from langchain.chains.combine_documents import create_stuff_documents_chain  # 문서들을 결합하는 체인 생성 도구
from langchain_openai import ChatOpenAI  # OpenAI의 LLM을 활용하는 도구
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama  # Ollama 모델을 사용하기 위한 도구
from langchain_upstage import UpstageEmbeddings  # Upstage에서 제공하는 임베딩 모델을 사용하기 위한 도구
from langchain_pinecone import PineconeVectorStore  # Pinecone의 벡터 스토어와 통합하기 위한 도구
from langchain_chroma import Chroma
# from langchain.llms import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.chat_message_histories import ChatMessageHistory  # 채팅 메시지 기록을 관리하는 클래스
from langchain_core.chat_history import BaseChatMessageHistory  # 기본적인 채팅 메시지 기록 클래스
from langchain_core.runnables.history import RunnableWithMessageHistory  # 메시지 기록과 함께 실행할 수 있는 클래스


from config import answer_examples  # 설정 파일에서 답변 예시를 불러옴

# 사용자 세션별로 채팅 기록을 저장하는 딕셔너리입니다.
store = {}

# 특정 세션 ID에 해당하는 채팅 기록을 가져오는 함수입니다. 해당 세션의 기록이 없다면 새로 생성합니다.
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()  # 세션 ID가 store에 없으면 새 채팅 기록을 생성하여 저장
    return store[session_id]


def get_retriever():
    # 'solar-embedding-1-large-query'라는 임베딩 모델을 사용하여 임베딩을 생성합니다.
    embedding = UpstageEmbeddings(model='solar-embedding-1-large-query')
    index_name = 'tax-markdown'  # 벡터 스토어의 인덱스 이름을 설정합니다.
    
    # 기존에 생성된 Pinecone 벡터 스토어 인덱스를 사용하여 데이터베이스를 설정합니다.
    # database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    database = Chroma(persist_directory='./chroma', collection_name='chroma-qna', embedding_function=embedding)
    
    # 설정된 데이터베이스를 검색기로 변환합니다. 여기서 k=4는 검색 결과의 상위 4개를 반환함을 의미합니다.
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever


# 과거 대화 내역을 고려한 검색기를 생성하는 함수입니다.
def get_history_retriever():
    llm = get_llm()  # 언어 모델(LLM)을 가져옵니다.
    retriever = get_retriever()  # 검색기를 가져옵니다.
    
    # 시스템 프롬프트: 과거 대화 내역을 고려하여 독립적으로 이해할 수 있는 질문을 생성하도록 유도하는 지침입니다.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # 사용자의 질문과 대화 내역을 결합하여 질문을 재구성하는 프롬프트 템플릿을 생성합니다.
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),  # 시스템의 지침 추가
            MessagesPlaceholder("chat_history"),  # 대화 기록을 위한 자리 표시자
            ("human", "{input}"),  # 사용자의 입력을 위한 자리 표시자
        ]
    )
    
    # 대화 내역을 고려한 검색기를 생성하여 반환합니다.
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever  # 과거 대화 기록을 고려하는 검색기 반환


# 사용하려는 LLM(언어 모델)을 가져오는 함수입니다.
def get_llm(model ='SOLAR-10.8B:latest'):
    # llm = ChatOpenAI(model='gpt-4o')
    # llm = Ollama(model='eeve:latest')
    # llm = ChatOllama(model='rokey:latest')
    # llm = Ollama(model=model)  # 현재 Ollama 모델을 사용하고 있으며, 기본값은 'rokey:latest'입니다.
    # llm = ChatOllama(model='bllosom-8b:latest')
    return llm


# 특정 사전 기반의 체인을 생성하는 함수입니다. 이 체인은 사전에 정의된 규칙에 따라 질문에 응답합니다.
def get_dictionary_chain():
    dictionary = ["너의 정체 -> 두산 로키 챗봇", "사용자가 질문하는 카드 -> 국민내일배움카드"]  # 사전 데이터: 특정 질문에 대한 답변
    llm = get_llm()  # 언어 모델(LLM)을 가져옵니다.
    
    # 사전 기반의 프롬프트 템플릿을 생성합니다. 
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 당신이 누군지 궁금해한다면, 두산 로키의 챗봇이라고 답해주세요.
        만약 당신을 궁금해하지 않다면, 당신의 정체를 나타낼 필요 없습니다. 그런 경우에는 질문만 리턴해주세요.
        문서에 관련 내용이 아니라면, 해당 부분에 대해서 답변해드리지 않는다고 말해주세요.
        사전: {dictionary}
        
        질문: {{question}}
    """)

    # 생성된 프롬프트 템플릿을 LLM과 결합하여 최종 출력을 문자열로 파싱하는 체인을 생성합니다.
    dictionary_chain = prompt | llm | StrOutputParser()
    
    return dictionary_chain  # 사전 기반 체인 반환


# RAG (Retrieval-Augmented Generation) 체인을 생성하는 함수입니다.
def get_rag_chain():
    llm = get_llm()  # 언어 모델(LLM)을 가져옵니다.

    # 예시를 기반으로 몇 가지 샘플 질문과 답변을 제공하는 프롬프트 템플릿을 생성합니다.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),  # 인간 사용자로부터 입력받은 질문
            ("ai", "{answer}"),  # AI가 제공한 답변
        ]
    )

    # 다수의 샘플 메시지를 포함한 Few-shot 프롬프트 템플릿을 생성합니다.
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,  # 사전에 정의된 답변 예시들
    )

    system_prompt = (
        "당신은 두산 교육과정 QnA 전문가입니다. 사용자의 QnA에 관한 질문에 답변해주세요"
        "아래에 제공된 문서에 기반해서 답변해주세요"
        "문서에 기반하여 답을 할 때, 사용자의 질문에 가장 적합한 부분을 활용해서 답변해주세요"
        "HRD-Net에 관련한 내용이 나온다면, HRD-Net의 주소는 https://www.work24.go.kr/cm/main.do 임을 참고해주세요."
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )
    
    # 시스템 프롬프트, Few-shot 프롬프트, 대화 기록을 결합한 최종 프롬프트 템플릿을 생성합니다.
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),  # 대화 기록
            ("human", "{input}"),  # 사용자 입력
        ]
    )

    # 과거 대화 내역을 고려한 검색기를 가져옵니다.
    history_aware_retriever = get_history_retriever()

    # 문서 검색과 QA를 결합한 체인을 생성합니다.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # RAG 체인을 생성하여 반환합니다. 이는 검색된 정보와 LLM을 결합하여 사용자의 질문에 답변합니다.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # RAG 체인을 대화 기록과 결합하여 최종 체인을 생성합니다.
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="input",  # 입력 메시지 키
        history_messages_key="chat_history",  # 대화 기록 메시지 키
        output_messages_key="answer",  # 출력 메시지 키
    ).pick('answer')  # 최종적으로 답변만을 반환하도록 설정
    
    return conversational_rag_chain  # 최종 대화 기반 RAG 체인 반환


# 사용자의 메시지를 받아 AI의 응답을 생성하는 함수
def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain()  # 사전 기반 체인을 가져옴
    rag_chain = get_rag_chain()  # RAG 체인을 가져옴

    # 입력 체인(사전 기반 체인)을 RAG 체인과 결합
    tax_chain = {"input": dictionary_chain} | rag_chain

    # 결합된 체인을 실행하여 AI의 응답을 스트리밍 방식으로 가져옴
    ai_response = tax_chain.stream(
        {
            "question": user_message  # 사용자의 질문
        },
        config={
            "configurable": {"session_id": "abc123"}  # 특정 세션 ID 설정
        },
    )

    # ai_response = tax_chain.stream(
    #     {
    #         "question": user_message  # 사용자의 질문
    #     }
    # )


    return ai_response  