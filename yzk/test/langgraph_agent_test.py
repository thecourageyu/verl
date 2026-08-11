#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import os

import datasets
import pandas as pd

# from verl.utils.hdfs_io import copy, makedirs
# import argparse


# In[2]:


from chat_model import SglangLLM


# In[3]:


from langgraph_agent_loop import ReactAgentLoop


# In[4]:


from transformers import AutoTokenizer


# In[5]:


model_path = "/mnt/yuzhe/models/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)


# In[6]:


# data_path = "/mnt/yuzhe/data/verl/gsm8k"
# data_path = "/mnt/yuzhe/data/verl/lighteval-MATH-preprocessed"
# data_path = "/mnt/yuzhe/data/verl/gsm8k-multiturn/"
data_path = "/mnt/yuzhe/data/verl/math_expression_tool/"


# In[7]:


df = pd.read_parquet(os.path.join(data_path, "test.parquet"))


# In[8]:


df


# In[9]:


df.iloc[12].prompt.tolist()


# In[10]:


df.iloc[12].prompt


# In[11]:


ra = ReactAgentLoop()


# In[12]:


raw_prompt = df.iloc[12].prompt


# In[13]:


sampling_params = {
    "temperature": 1.0,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}


# In[14]:


result = ra.run(sampling_params, raw_prompt=raw_prompt, tokenizer=tokenizer)


# In[ ]:


a = await ra.run(sampling_params, raw_prompt=raw_prompt, tokenizer=tokenizer)


# In[ ]:




# In[ ]:


if __name__ == "__main__":
    # find the sglang server port
    # ss -tulpn | grep sgl

    import openai
    port = 37335
    port = 31654
    sgl_llm = SglangLLM(
        client=openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
    )

    from langchain_core.messages import BaseMessage
    from langgraph.graph import END, MessagesState, StateGraph
    from langgraph.graph.message import add_messages

    # This is roughly what it looks like under the hood:
    from typing import Annotated, TypedDict, Sequence

    class MessagesState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]


    # def call_model(state: State):
    def call_model(state):
        # print(state, " <<<<<<<<<<<< state 123\n\n\n")
        response = sgl_llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)

    app = graph.compile()
    result = app.invoke({"messages": df.iloc[12].prompt.tolist()})
    # result = app.invoke({"messages": [{"role": "user", "content": "你老師好，請問如何使用 tool call?"}]})
    # result = app.invoke([{"role": "user", "content": "你好"}])

    print(result)
    # response = sgl_llm..invoke(state["messages"])
    # return {"messages": state["messages"] + [response]}

