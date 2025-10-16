from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from host.tools import add, sub, multiply, divide
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import httpx
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
import streamlit as st
from RAG_test import sql_agent, waste_management_QA, waste_management_details

client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-gpt-35-turbo",
    api_key="sk-Ycj1VI3lPw9Wn_qAFtcFvA",  # Replace with your actual API key
    http_client=client
)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

sys_prompt_for_chef = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''
            You are Food Menu planning assistant to assist Chef to plan dishes, menu plans, ingredient utilisation
            INSTRUCTIONS
            1. Provide maximum number of results unless the user specifies the limit
            2. Ask question interactively if you need further information
            3. Provide step wise clear description on how you reason for the response
            Use the following tools if required 
            1. sql_agent - Provide details about recipes, cuisine, ingredient, cook time, instructions.
            ''',
        ),
        ("placeholder", "{messages}"),
    ]
)

sys_prompt_for_manager = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''
            You are Food Menu planning assistant to assist Hotel manger to menu plans, control waste management and food waste details
            INSTRUCTIONS
            1. Provide maximum number of results unless the user specifies the limit
            2. Ask question interactively if you need further information
            3. Provide step wise clear description on how you reason for the response
            Use the following tools if required 
            1. waste_management_details - Provide details about food waste risk details and recommended strategies
            2. waste_management_QA - Provides comprehensive Food Waste Reduction Strategies for Multi-Cuisine Hotels
            ''',
        ),
        ("placeholder", "{messages}"),
    ]
)


with st.sidebar:
    persona = st.selectbox(
    "Choose a persona",
    ("Chef", "Manager"),
    key="persona_select"
    )

if persona == "Chef":
    prompt = sys_prompt_for_chef
    tools = [sql_agent]
elif persona == "Manager":
    prompt = sys_prompt_for_manager
    tools = [waste_management_QA, waste_management_details]
llm_with_tools = prompt | llm.bind_tools(tools)
def chatbot(state: State):
    messages = trim_messages(
        state['messages'],
        strategy = "last",
        token_counter = count_tokens_approximately,
        max_tokens = 100000
    )
    state['messages'] = llm_with_tools.invoke({"messages":messages})
    return state

builder = StateGraph(State)
builder.add_node("chat_node", chatbot)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chat_node")  # Start with the assistant
builder.add_conditional_edges("chat_node", tools_condition)  # Move to tools after input
builder.add_edge("tools", "chat_node")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {
    "configurable": {
        "thread_id": 1234,
    }
}

chat_map = {'human':'user', 'ai': 'assistant'}

st.title("AI Food Menu Planning Assistant")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(chat_map[msg.type]):
        st.markdown(msg.content)

if prompt := st.chat_input("Ask you question ...", key = 'user_input'):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(content = prompt))
    with st.spinner("Generating answer ...", show_time = True):
        response = graph.invoke({'messages':st.session_state.messages}, config = config)
    with st.chat_message("assistant"):
        st.session_state.messages.append(AIMessage(content = response["messages"][-1].content))
        st.markdown(response["messages"][-1].content)
