from langgraph.graph import MessagesState, StateGraph, START,END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import (SystemMessage, AIMessage ,HumanMessage ,RemoveMessage)
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
from typing import TypeDict , Annotated
import operator

load_dotenv()

llm=ChatOpenAI(model_name="deepseek-chat",temperature=0,max_tokens=2000,api_key=os.getenv("deepseek_api_key"),base_url=os.getenv("LLM_BASE_URL"))

@tool
def add(a:int ,b:int)  ->int:
    return a+b


@tool
def multiply(a:int,b:int) ->int:
    return a*b

@tool
def divide(a:int , b:int) ->int:
    return a/b

tools=[add, multiply,divide]
llm_with_tools=llm.bind_tools(tools)
sys_msg=SystemMessage(content="you are a helpful assistant")

def assistant(state:MessagesState):
    return {"messages":[llm_with_tools.invoke([sys_msg]+state["messages"])]}

def build_react_graph():
  builder = StateGraph(MessagesState)
  builder.add_node("assistant", assistant)
  builder.add_node("tools",ToolNode(tools))
  builder.add_edge(START,"assistant")
  builder.add_conditional_edges("assistant",tools_condition)
  builder.add_edge("tools","assistant")
  return builder
builder = build_react_graph()

class CustomState(TypeDict):
    messages: Annotated[list, "messages"]
    docs:Annotated[list,operator.add]
    counter:int

def delete_old_messages(state:CustomState):
    return{
        "messages": [
                   RemoveMessage(id=m.id)
                   for m in state["messages"][:-2]
               ]
    }

memory=MemorySaver()
graph=builder.compile(checkpointer=memory)
graph.invoke({"messages":[...]},{"configurable":{"thread_id":"text"}})