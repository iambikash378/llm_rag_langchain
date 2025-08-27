from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

# add_routes(
#     app,
#     ChatGroq(),
#     path='/groq',
# )

deepseek = ChatGroq(
    model = "deepseek-r1-distill-llama-70b"
)

llama = ChatGroq(
    model = "llama-3.3-70b-versatile"
)

prompt1 = ChatPromptTemplate.from_template("Write an essay about {topic} with 200 words")
prompt2 = ChatPromptTemplate.from_template("Write a poem about {topic} with 200 words")

add_routes(
    app,
    prompt1|deepseek,
    path="/essay"
)

add_routes(
    app,
    prompt2 | llama,
    path="/poem" 
)

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port = 8000)