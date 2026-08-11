# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
LangGraph React Agent Loop.

This implementation is exact same as `ToolAgentLoop`.

Ref: https://langchain-ai.github.io/langgraph/tutorials/workflows/
"""

import sys
print(sys.path)

import logging
from typing import Any, Literal

# from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
# from recipe.langgraph_agent.chat_model import (
# from yzk.test.chat_model import (
from chat_model import (
    # ChatModel,
    SglangLLM,
    MaxTokenExceededError,
    # convert_to_agent_output,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.tools import tool

import openai
# from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

logger = logging.getLogger(__name__)



# def convert_to_agent_output(messages: list[BaseMessage], response_length: int) -> AgentLoopOutput:
def convert_to_agent_output(messages: list[BaseMessage], response_length: int):
    """Convert messages to AgentLoopOutput.

    Args:
        messages (List[BaseMessage]): List of messages, last message must be assistant
            with response_metadata containing `prompt_ids` and `response_mask`.
        response_length (int): Max length of response.

    Returns:
        AgentLoopOutput: agent loop output trajectory used for training.
    """

    # print(f"[CheckFormat][AgentLoopOutput]\nmessages: {messages} ({type(messages)})\nresponse_length: {response_length}\n\n")
    # logger.warning(f"[CheckFormat][AgentLoopOutput]\nmessages: {messages} ({type(messages)})\nresponse_length: {response_length}\n\n")


    # skip last tool calls
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type != "tool":
            break
    last_message = messages[i]
    assert last_message.type == "ai", f"Last message must be assistant, but got {last_message.type}"
    assert "prompt_ids" in last_message.response_metadata, "Last message must have prompt_ids in response_metadata"
    assert "response_mask" in last_message.response_metadata, (
        "Last message must have response_mask in response_metadata"
    )

    num_turns = 0
    for i in range(len(messages)):
        if messages[i].type == "system":
            continue
        # parallel tool calls are in single turn
        if i == 0 or messages[i].type != messages[i - 1].type:
            num_turns += 1

    prompt_ids = last_message.response_metadata["prompt_ids"]
    response_mask = last_message.response_metadata["response_mask"]

    response_ids = prompt_ids[-len(response_mask) :]
    prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

    
    # assert len(response_ids) > 0
    # logger.warning(f"[CheckFormat][AgentLoopOutput]\nresponse_ids: {response_ids} ({type(response_ids)})\nprompt_ids: {prompt_ids} ({type(prompt_ids)})\n\n")

    # output = AgentLoopOutput(
    #     prompt_ids=prompt_ids,
    #     response_ids=response_ids[:response_length],
    #     response_mask=response_mask[:response_length],
    #     # prompt_ids=torch.tensor(prompt_ids),
    #     # response_ids=torch.tensor(response_ids[:response_length]),
    #     # response_mask=torch.tensor(response_mask[:response_length]),
    #     num_turns=num_turns,
    #     metrics={},
    # )

    output = {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids[:response_length],
        "response_mask": response_mask[:response_length],
        "num_turns": num_turns,
        "metrics": {},
    }
    return output

    
@tool(parse_docstring=True)
def calculate(a: int, b: int, operand: str) -> int:
    """
    Compute the results using operand with two integers

    Args:
        a: the first operand
        b: the second operand
        operand: '+' or '-' or '*' or '@'
    """
    assert operand in ["+", "-", "*", "@"], f"unknown operand {operand}"
    if operand == "@":
        return 3 * a - 2 * b
    return eval(f"{a} {operand} {b}")


# async def call_model(state: MessagesState, config: RunnableConfig):
def call_model(state: MessagesState, config: RunnableConfig):

    # 這個 model 是外部實例化的 langgraph 但主要依靠 verl.workers.rollout.llm_server.LLMServerClient 產生 rollout
    model = config["configurable"]["model"]
    # print(f"[ReactAgent][call_model]\n\t[model] {model} ({type(model)})\n\t[config]\n{config}\n")
    sampling_params = config["configurable"]["sampling_params"]
    try:
        # print(f"[call_model][state] {state}\n\n")
        # message = await model.ainvoke(state["messages"], sampling_params=sampling_params)
        message = model.invoke(state["messages"], sampling_params=sampling_params)
        # print(f"[call_model][messages] {message}\n\n")
        return {"messages": [message]}
    except MaxTokenExceededError:
        # last message is ToolMessage
        return {"messages": []}


def should_continue(state: MessagesState, config: RunnableConfig) -> Literal["tools", END]:
    # Safely extract max_assistant_turns from config
    max_assistant_turns = None
    try:
        if config and "configurable" in config:
            max_assistant_turns = config["configurable"].get("max_assistant_turns")
    except Exception as e:
        logger.warning(f"Failed to extract max_assistant_turns from config: {e}")

    num_assistant_turns = 0
    for message in state["messages"]:
        if message.type == "ai":
            num_assistant_turns += 1

    last_message = state["messages"][-1]

    # LLM call failed, e.g: max response length exceeded
    if last_message.type == "tool":
        return END

    # max assistant turns exceeded
    # Use a reasonable default limit (25) if max_assistant_turns is not set
    # This prevents infinite loops
    effective_max_turns = max_assistant_turns if max_assistant_turns is not None else 25
    if num_assistant_turns >= effective_max_turns:
        return END

    # no tool calls
    if not getattr(last_message, "tool_calls", None):
        return END

    return "tools"


# class ReactAgentLoop(AgentLoopBase):
class ReactAgentLoop:
    # Recursion limit calculation constants
    DEFAULT_MAX_ASSISTANT_TURNS = 25
    MIN_RECURSION_LIMIT = 50
    NODES_PER_TURN = 2  # Each AI turn involves agent + tools nodes
    RECURSION_LIMIT_SAFETY_FACTOR = 1.5  # 50% buffer for edge cases

    # def __init__(self, *args, **kwargs):
    # def __init__(self, args):
    def __init__(self):
        # print("[ReactAgentLoop] __init__")
        # super().__init__(*args, **kwargs)
        self.tools = [calculate]
        self.graph = self.build_graph()


    def build_graph(self) -> StateGraph:
        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END,
            },
        )

        workflow.add_edge("tools", "agent")
        graph = workflow.compile()
        return graph

    
    # async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
    # async def run(self, sampling_params: dict[str, Any], **kwargs):
    def run(self, sampling_params: dict[str, Any], **kwargs):
        messages = list(kwargs["raw_prompt"])

        
        print(f"[ReactAgentLoop] run\n[raw_prompt] {messages} ({len(messages)})\n")
        
        tokenizer = kwargs["tokenizer"]

        port = 37335
        port = 31654
        model = SglangLLM(
            client=openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None"),
            tokenizer=tokenizer,
        )
        
        # model_path = self.config.actor_rollout_ref.model.path
        # model_name = "/".join(model_path.split("/")[-2:])

        # rollout = self.config.actor_rollout_ref.rollout
        # The BaseChatModel of LangChain. It will call the object of LLMServerClient to get the response
        # model = ChatModel(
        #     model=model_name,
        #     client=self.server_manager,
        #     tokenizer=self.tokenizer,
        #     max_tokens=rollout.response_length,
        #     max_parallel_calls=rollout.multi_turn.max_parallel_calls,
        #     tool_parser=rollout.multi_turn.format,
        # )

        # model.bind_tools 必須重新賦值給變數或覆蓋掉原變數, 得到一個 RunnableBinding (model_with_tools)
        # 此時 model._generate or model._agenerae 可以用 kwargs['tools'] 取出 tools
        model = model.bind_tools(self.tools, tool_choice="any")

    #     # Calculate recursion_limit dynamically based on max_assistant_turns
    #     max_assistant_turns = (
    #         rollout.multi_turn.max_assistant_turns
    #         if rollout.multi_turn.max_assistant_turns
    #         else self.DEFAULT_MAX_ASSISTANT_TURNS
    #     )

    #     # Formula: nodes_per_turn * max_turns * safety_buffer, with minimum threshold
    #     recursion_limit = max(
    #         self.MIN_RECURSION_LIMIT,
    #         int(max_assistant_turns * self.NODES_PER_TURN * self.RECURSION_LIMIT_SAFETY_FACTOR),
    #     )
    #     logger.info(f"Configured recursion_limit={recursion_limit} (max_assistant_turns={max_assistant_turns})")

        config = {
            "configurable": {
                "model": model,
                "sampling_params": sampling_params,
                # "max_user_turns": rollout.multi_turn.max_user_turns,
                # "max_assistant_turns": rollout.multi_turn.max_assistant_turns,
                "max_user_turns": 5,
                "max_assistant_turns": 10,
            },
            # "recursion_limit": recursion_limit,
        }

    #     # print(f"[react_agent_loop][run] messages: {messages}")
        
    #     # TODO: how to handle multiple trajectories in an graph invocation?
    #     # Each graph node may has its own LLM calls and state, e.g:
    #     # https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart
        try:
            # state = await self.graph.ainvoke(input={"messages": messages}, config=config)
            state = self.graph.invoke(input={"messages": messages}, config=config)
        except Exception as e:
            logger.error(f"Agent loop execution failed: {type(e).__name__}: {e}")
            logger.error("Falling back to a minimal dummy trajectory.")

            # Fallback to a minimal assistant message so that
            # convert_to_agent_output and downstream padding logic
            # can still run without crashing.
            # dummy_id = 0
            # fallback_message = AIMessage(
            #     content="[Agent execution failed - no valid trajectory]",
            #     response_metadata={
            #         "request_id": "fallback",
            #         # "request_id": -9999,
            #         "prompt_ids": [dummy_id, dummy_id],
            #         "response_mask": [1],  # 原來
            #         # "response_mask": [0, 1],
            #     },
            # )
            # state = {"messages": [fallback_message]}

        output = state
        print(output, " <<< check output\n")
        # output = convert_to_agent_output(state["messages"], rollout.response_length)
        # output = convert_to_agent_output(state["messages"], 512)

        # print(f"[ReactAgentLoop] output = {output}\n")
        return output




if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="What's wrong with you")
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--epochs", type=int, default=10)
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    # python script.py train --epochs 20
    # python script.py eval --checkpoint model.pt
    
    
    parser.add_argument("--epochs", type=int, default=10, required=True)       # 必填參數
    

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--name", type=str, required=True, help="實驗名稱")
    
    parser.add_argument("--gpu", action="store_true")              # flag,出現就是 True
    parser.add_argument("--model", choices=["resnet", "vgg", "cnn"], default="resnet")  # 限定選項
    parser.add_argument("--layers", type=int, nargs="+")           # 接受多個值,list
    parser.add_argument("-o", "--output", type=str)                # 同時支援短、長參數名
    args = parser.parse_args()
    
    print(args.lr)
    print(args.batch_size)
    print(args.name)
    
    # python train.py --lr 0.01 --batch_size 64 --name exp1
    
    agent_loop = ReactAgentLoop(args)