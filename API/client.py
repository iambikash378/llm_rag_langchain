import requests 
import streamlit as st 

def get_deepseek_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json={ 'input' : {'topic': input_text}}
    )

    return response.json()['output']['content']


def get_llama_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json={'input': {'topic': input_text}}
    ) 

    return response.json()['output']['topic']


st.title("Langchain Demo with Groq (2 Models)")
input_text1 = st.text_input("Write an essay on")
input_text2= st.text_input("Write a poem on")

if input_text1:
    st.write(get_deepseek_response(input_text1))

if input_text2:
    st.write(get_llama_response(input_text2))



