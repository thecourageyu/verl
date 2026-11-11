"""陪伴機器人用的兩個 verl 工具(純 Python、記憶體內、每條軌跡獨立狀態,不需要沙盒)。

- MemorySearchTool : search_memory     唯讀檢索,資料來自 dataset 的 tools_kwargs
- ReminderTool     : manage_reminder   有狀態(pending 清單),用來訓練「先查 pending、再確認、才建立」

注意:不同 verl 版本 create() 收到參數的形式略有差異(攤平 kwargs 或包在 create_kwargs 裡),
下方 _get() 兩種都接。請以你安裝版本的 verl/tools/base_tool.py 為準。
"""
import re
import uuid
from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


def _as_list(x) -> list:
    if x is None:
        return []
    return [str(i) for i in list(x)]  # parquet 讀回來可能是 numpy array


def _get(kwargs: dict, key: str, default=None):
    if kwargs.get(key) is not None:
        return kwargs[key]
    return (kwargs.get("create_kwargs") or {}).get(key, default)


def _grams(text: str) -> set:
    text = re.sub(r"\s+", "", str(text).lower())
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}  # 中文用 char bigram,免斷詞


class MemorySearchTool(BaseTool):
    """search_memory(query):回傳最相關的 top_k 筆真實記憶;找不到要明確回報『沒有』。"""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.top_k = int(config.get("top_k", 3))
        self.min_overlap = float(config.get("min_overlap", 0.15))
        self._mem: dict[str, list[str]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid.uuid4())
        self._mem[instance_id] = _as_list(_get(kwargs, "memories", []))
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = str(parameters.get("query", "")).strip()
        if not query:
            return ToolResponse(text="錯誤:query 不可為空,請提供要查詢的關鍵字,例如「主管」「面試」。"), 0.0, {"error": "empty_query"}
        qg = _grams(query)
        scored = []
        for m in self._mem.get(instance_id, []):
            ov = len(qg & _grams(m)) / max(1, len(qg))
            if ov >= self.min_overlap:
                scored.append((ov, m))
        scored.sort(key=lambda t: -t[0])
        hits = [m for _, m in scored[: self.top_k]]
        if not hits:
            return ToolResponse(text="沒有找到相關記憶。請不要假設使用者說過這件事,可以自然地請對方多說一點。"), 0.0, {"hits": 0}
        return ToolResponse(text="找到的記憶:\n" + "\n".join(f"- {h}" for h in hits)), 0.0, {"hits": len(hits)}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._mem.pop(instance_id, None)


class ReminderTool(BaseTool):
    """manage_reminder(action=list|create, ...):
    工具本身就會對違規操作回傳錯誤訊息,讓模型在 rollout 內「試錯」並學到正確流程:
      1) create 前必須先 list;2) 必須 user_confirmed=true 才會建立。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._state: dict[str, dict] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid.uuid4())
        self._state[instance_id] = {"pending": _as_list(_get(kwargs, "pending", [])), "listed": False}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        st = self._state.get(instance_id)
        if st is None:
            return ToolResponse(text="錯誤:工具狀態不存在。"), 0.0, {"error": "no_state"}
        action = str(parameters.get("action", "")).strip().lower()

        if action == "list":
            st["listed"] = True
            if not st["pending"]:
                return ToolResponse(text="目前沒有 pending 的提醒。"), 0.0, {"pending": 0}
            return ToolResponse(text="目前 pending 的提醒:\n" + "\n".join(f"- {p}" for p in st["pending"])), 0.0, {"pending": len(st["pending"])}

        if action == "create":
            if not st["listed"]:
                return ToolResponse(text="錯誤:建立前必須先呼叫 action=list 檢查是否已有 pending 提醒。"), 0.0, {"error": "not_listed"}
            if not parameters.get("user_confirmed"):
                return ToolResponse(text="錯誤:使用者尚未確認。請先用自然語言向使用者確認時間與內容,等對方同意後再建立。"), 0.0, {"error": "not_confirmed"}
            time_, content = str(parameters.get("time", "")).strip(), str(parameters.get("content", "")).strip()
            if not time_ or not content:
                return ToolResponse(text="錯誤:time 與 content 都必須提供。"), 0.0, {"error": "missing_fields"}
            item = f"{time_} {content}"
            if item in st["pending"]:
                return ToolResponse(text="已經有相同的提醒,不需要重複建立。"), 0.0, {"error": "duplicate"}
            st["pending"].append(item)
            return ToolResponse(text=f"已建立提醒:{item}"), 0.0, {"created": 1}

        return ToolResponse(text="錯誤:action 只能是 list 或 create。"), 0.0, {"error": "bad_action"}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._state.pop(instance_id, None)


class Memory(BaseTool):
    """memory(operation, query): 依用戶需求調整記憶，包含 create, ignore, read, update, forget 等操作""",

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.top_k = int(config.get("top_k", 3))
        self.min_overlap = float(config.get("min_overlap", 0.15))
        self._mem: dict[str, list[str]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid.uuid4())
        self._mem[instance_id] = _as_list(_get(kwargs, "memories", []))
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        operation = str(parameters.get("operation", "")).strip()
        query = str(parameters.get("query", "")).strip()
        if not operation:
            return ToolResponse(text="[ERR] operation 不可為空，請提供要查詢的關鍵字，例如 create, read 或 update 等。"), 0.0, {"error": "empty_operation"}
#        qg = _grams(query)
#        scored = []
#        for m in self._mem.get(instance_id, []):
#            ov = len(qg & _grams(m)) / max(1, len(qg))
#            if ov >= self.min_overlap:
#                scored.append((ov, m))
#        scored.sort(key=lambda t: -t[0])
#        hits = [m for _, m in scored[: self.top_k]]
#        if not hits:
#            return ToolResponse(text="沒有找到相關記憶。請不要假設使用者說過這件事,可以自然地請對方多說一點。"), 0.0, {"hits": 0}
        return ToolResponse(text="操作完成"), 0.0, {"hits": 0}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._mem.pop(instance_id, None)
