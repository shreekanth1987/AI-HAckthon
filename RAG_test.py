import pandas as pd
import httpx
from pathlib import Path
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://easyjet-ejdp-data-dev.cloud.databricks.com/serving-endpoint",
    model="databricks-gpt-oss-20b",
    api_key="",  
    http_client=client
)

embedding_model = OpenAIEmbeddings(
    base_url="https://easyjet-ejdp-data-dev.cloud.databricks.com/serving-endpoint",
    model="databricks-gpt-oss-20b",
    api_key="",  
    http_client=client
)



# SQL Database setup
engine = create_engine("sqlite:///Food_Recipe_with_season.db")
db = SQLDatabase(engine=engine)
agent_executor = create_sql_agent(
    llm, 
    db=db, 
    agent_type="openai-tools", 
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

engine_for_waste_management = create_engine("sqlite:///Waste_Reduction_Matrix.db")
db_for_waste_management = SQLDatabase(engine=engine_for_waste_management)
agent_executor_for_waste_management = create_sql_agent(
    llm, 
    db=db_for_waste_management, 
    agent_type="openai-tools", 
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# Initialize vectorstore once
def _load_or_create_vectorstore():
    vectorstore_path = "waste_management_faiss_index"
    
    if Path(vectorstore_path).exists():
        print("📂 Loading existing FAISS vectorstore...")
        return FAISS.load_local(
            vectorstore_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        print("🔨 Creating new FAISS vectorstore...")
        loader = PyPDFLoader("Food_Waste_Reduction_Strategies_MultiCuisine_Hotel.pdf")
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        texts = text_splitter.split_documents(pages)
        
        vectorstore = FAISS.from_documents(texts, embedding_model)
        vectorstore.save_local(vectorstore_path)
        print("✅ FAISS vectorstore created and saved")
        
        return vectorstore

# Global vectorstore
_VECTORSTORE_CACHE = _load_or_create_vectorstore()
_RETRIEVER = _VECTORSTORE_CACHE.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# QA prompt template
_QA_PROMPT = ChatPromptTemplate.from_template("""Use the following context to answer the question about food waste reduction strategies.

Context: {context}

Question: {question}

Provide a detailed answer based on the context above:""")

# LCEL chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

_QA_CHAIN = (
    {"context": _RETRIEVER | format_docs, "question": RunnablePassthrough()}
    | _QA_PROMPT
    | llm
    | StrOutputParser()
)

@tool
def sql_agent(query: str) -> str:
    '''
    Provide details about recipes, cuisine, ingredient, cook time, instructions
    '''
    return agent_executor.invoke({"input": query})

@tool
def waste_management_QA(query: str) -> str:
    '''
    Provides comprehensive Food Waste Reduction Strategies for Multi-Cuisine Hotels
    '''
    try:
        result = _QA_CHAIN.invoke(query)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def waste_management_details(query: str) -> str:
    '''
    Provide details about food waste risk details and recommended strategies
    '''
    return agent_executor_for_waste_management.invoke({"input": query})
