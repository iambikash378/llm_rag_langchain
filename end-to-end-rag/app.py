import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain 
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(
        model='all-minilm'
    )
    st.session_state.loader = WebBaseLoader('https://docs.langchain.com/langsmith/observability-concepts')
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    st.session_state.final_docs = st.session_state.text_splitter.split_documents(
         st.session_state.docs,
    )

    st.session_state.vectors = FAISS.from_documents(
    documents = st.session_state.final_docs,
    embedding=st.session_state.embeddings
    )

st.title("ChatGroq Demo")
llm = ChatGroq(model = 'deepseek-r1-distill-llama-70b', api_key=GROQ_API_KEY)
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only. Provide the most accurate response based on the context.

<context>
{context}
</context>

Questions : {input}

"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here")

if prompt:
    start_time = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response time: ", time.process_time() - start_time)
    st.write(response['answer']) 

    with st.expander("Document Similarity Search from Vector DB"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------")