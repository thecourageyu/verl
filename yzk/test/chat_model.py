
import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages import convert_to_openai_messages
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import Any, List, Optional
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.language_models.base import LanguageModelInput


class MaxTokenExceededError(Exception):
    """Indicate that history chat messages + tool message exceeds LLM max_tokens."""
    pass

class SglangLLM(BaseChatModel):
    client: Any

    tokenizer: Any
    """Tokenizer for the model"""

    max_tokens: int = 256
    """Max tokens to generate"""

    tool_parser: str = "hermes"
    """Tool parser for the model"""

    max_parallel_calls: int = 1
    """Max parallel tool calls"""

    temperature: float = 1.0
    """Temperature for sampling"""

    top_p: float = 1.0
    """Top p for sampling"""

    repetition_penalty: float = 1.0
    """Repetition penalty for sampling"""

    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:

        # 把 LangChain messages 轉成你自己 API 需要的格式
        # prompt = "\n".join([f"{m.type}: {m.content}" for m in messages])

        # SglangLLM 物件, 如 model = SglangLLM(...), 
        # 呼叫 bind_tools 後, 重新賦值給另一變數, 如 new_model = model.bind_tools(...) 
        # new_model 將得到一個 RunnableBinding (model_with_tools) 
        # 此時 new_model._generate or new_model._agenerae 可以用 kwargs['tools'] 取出 tools
        
        tools_schema = kwargs.get("tools", None)

        print(f"[SglangLLM] run\n[tools_schema] {tools_schema} ({len(tools_schema)}) {type(tools_schema)}\n")
        tools_json_str = json.dumps(tools_schema, ensure_ascii=False)
        print(f"[SglangLLM] run\n[tools_json_str] {tools_json_str} ({len(tools_json_str)}) {type(tools_json_str)}\n")

        # _tools_json_str = ""
        # for t in tools_json_str:
        #     _tools_json_str += f"{t}\n"
        
        system_prompt = {
            "role": "system", 
            "content": (
                "You have access to the following tools:\n\n"
                f"{tools_json_str}\n"
                "If you need to call a tool, respond ONLY with JSON:\n"
                '{"action": "<tool_name>", "action_input": {<args>}}'
            )
        }
        print(f"[SglangLLM] run\n[tools] {tools} ({len(tools)})\n")
        print(f"[SglangLLM] run\n[system_prompt] {system_prompt} ({len(system_prompt)})\n")
        

        
        # 呼叫你自己的模型 (可以是 requests.post、本地模型推論等等)
        # response_text = self._call_my_api(prompt)
        response_text = self._call_openai_api(messages, system_prompt)

        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:

        # 把 LangChain messages 轉成你自己 API 需要的格式
        # prompt = "\n".join([f"{m.type}: {m.content}" for m in messages])
        
        # 呼叫你自己的模型 (可以是 requests.post、本地模型推論等等)
        # response_text = self._call_my_api(prompt)
        response_text = self._call_openai_api(messages)

        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def bind_tools(self, tools, **kwargs) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the model.

        Args:
            tools: Sequence of tools to bind to the model.

        Returns:
            A Runnable that returns a message.
        """
        formatted_tools: list = [convert_to_openai_tool(tool) for tool in tools]
        print(f"[formatted_tools] {formatted_tools}\n")
        
        # used to remove system prompt prefix when encoding tool response
        system_prompt = self.tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
        kwargs["system_prompt"] = system_prompt

        print(f"[system_prompt] {system_prompt}\n")

        return self.bind(tools=formatted_tools, **kwargs)
    
    def _call_openai_api(self, messages: List[BaseMessage], system_prompt: str = None) -> str:
        # import requests
        # resp = requests.post(self.api_url, json={"prompt": prompt})
        # return resp.json()["text"]

        # print(f"self.tools: {self.tools}\n")
        
        openai_messages = convert_to_openai_messages(messages)

        if system_prompt is not None:
            openai_messages.insert(0, system_prompt)
        print(f"openai_messages: {openai_messages}\n")
        
        response = self.client.chat.completions.create(
            model="qwen2.5-3b-500",
            # messages=[
            #     {"role": "user", "content": "List 3 countries and their capitals."},
            # ],
            messages=openai_messages,
            # temperature=1.0,
            temperature=0.7,
            max_tokens=256,
        )

        # return response.json()["text"]
        return response.choices[0].message.content
    
    @property
    def _llm_type(self) -> str:
        return "sglang-llm"

    # def close_client(self):
    #     self.client.close()


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
    result = app.invoke({"messages": [{"role": "user", "content": "你老師好，請問如何使用 tool call?"}]})
    # result = app.invoke([{"role": "user", "content": "你好"}])

    print(result)
    # response = sgl_llm..invoke(state["messages"])
    # return {"messages": state["messages"] + [response]}