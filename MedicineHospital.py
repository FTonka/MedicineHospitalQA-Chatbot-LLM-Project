import os
import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch


os.environ["OPENAI_API_KEY"] = 'api-key'

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def load_pdf_and_create_answer(question, chat_history):
    loader = PyPDFLoader("PDFs/MergedMedicine.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(texts, embeddings)

    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 3})

    chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, verbose=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=retriever,
        verbose=True
    )

    res = chain({"question": question, "chat_history": chat_history})

    return res

def main():
    st.title("Medicine Hospital")
    query = st.text_input("Merhaba, nasıl yardımcı olabilirim?")
    if st.button("Sor"):
        response = ask_question(query)
        st.session_state.chat_history.append((query, response))
        st.write("Cevap:", response)
        st.write("Chat Geçmişi:", st.session_state.chat_history)

@st.cache_data
def ask_question(query):
    result = load_pdf_and_create_answer(query, st.session_state.chat_history)
    return result["answer"]


if __name__ == "__main__":
    main()
