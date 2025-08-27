# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY") # For langsmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the queries"),
        ("user", "Question:{question}")
    ]
)

#Stremlit framework

st.title("Langchain Demo with OpenAPI")
input_text = st.text_input("Query here")

# OpenAI LLMs

llm = ChatGroq(
    model = "deepseek-r1-distill-llama-70b"
)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))
