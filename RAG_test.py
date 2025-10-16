from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import pandas as pd
from langchain_community.agent_toolkits import create_sql_agent
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-gpt-35-turbo",
    api_key="sk-Ycj1VI3lPw9Wn_qAFtcFvA",  # Replace with your actual API key
    http_client=client
)

embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-2hYRw_e4aDq5ELK4xHX3gA",  # Replace with your actual API key
    http_client=client
)

import os

tiktoken_cache_dir = "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
# df = pd.read_csv("Food_Recipe.csv")


engine = create_engine("sqlite:///Food_Recipe_with_season.db")
db = SQLDatabase(engine=engine)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

engine_for_waste_management = create_engine("sqlite:///Waste_Reduction_Matrix.db")
db_for_waste_management = SQLDatabase(engine=engine_for_waste_management)
agent_executor_for_waste_management = create_sql_agent(llm, db=db_for_waste_management, agent_type="openai-tools", verbose=True)

@tool
def sql_agent(query:str)->str:
    '''
    Provide details about recipes, cuisine, ingredient, cook time, instructions
    Args:
    query - User query
    Returns:
    returns the response for the user query based on the information available in the table in a json output
    '''
    return agent_executor.invoke({"input": query})

@tool
def waste_management_QA(query:str):
    '''
    Provides comprehensive Food Waste Reduction Strategies for Multi-Cuisine Hotels
    Args:
    query - User query
    returns:
    Response to the user query in json format like topic:description

    '''
    loader = PyPDFLoader("Food_Waste_Reduction_Strategies_MultiCuisine_Hotel.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    vectorstore = FAISS.from_documents(texts, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    result = qa_chain.invoke(query)
    return result

@tool
def waste_management_details(query:str)->str:
    '''
    Provide details about food waste risk details and recommended strategies
    Args:
    query - User query
    Returns:
    returns the response for the user query based on the information available in the table in json format
    '''
    return agent_executor_for_waste_management.invoke({"input": query})