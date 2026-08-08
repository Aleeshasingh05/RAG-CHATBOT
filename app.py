import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def load_chain():
    loader = PyPDFLoader("Impact of AI.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

rag_chain = load_chain()

st.title("RAG Application using Gemini")
query = st.chat_input("Ask me anything:")

if query:
    response = rag_chain.invoke({"input": query})
    with st.expander("Retrieved context"):
        for doc in response["context"]:
            st.write(doc.page_content[:300])
    st.write(response["answer"])