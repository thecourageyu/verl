Q1. 具備思考能力的 LLM 即使關閉思考，訓練語料是否得加 <think>\n\n</think>\n\n

(1) 推論時:歷史 assistant 輪不用加,也不該加

Qwen3 官方的建議是:多輪對話中,歷史的模型輸出只保留最終回答,不放 thinking 內容。
Qwen3 的 chat template 本身就實作了這點,在最後一個 user 提問之前的 assistant 輪,會自動把 <think>...</think> 剝掉。

所以推論時:

歷史 assistant 的 content 放純回答即可。就算你放了空 think,template 也會幫你剝掉。
要關閉 thinking,用 enable_thinking=False,template 會自己在 generation prompt 結尾補上 <think>\n\n</think>\n\n,不要手動再加。
用 vLLM 等 OpenAI 相容 API 時,透過 chat_template_kwargs={"enable_thinking": False} 傳入。

(2) 訓練時:沒有一個「每輪都要加」的標準答案

原則是訓練時的格式要貼近推論時的格式。推論時,「正在生成的那一輪」前面一定有空 think,歷史輪沒有。所以理論上:

|  輪次                       |  理想格式                        |
------------------------------------------------------------------
|  被訓練的目標 assistant 輪  |  前面有 <think>\n\n</think>\n\n  |
|  作為歷史的 assistant 輪    |  沒有 think                      |

問題在於,多輪資料塞成一條序列、每個 assistant 輪都算 loss 時,中間的輪次既是 "目標又是歷史" ,沒辦法同時滿足。常見的取捨有三種:

- 只在最後一輪算 loss:格式完全符合 template,和推論一致,但資料利用率低。
- 拆成多筆:每個 assistant 輪各成一筆(前面的輪次當歷史、不帶 think,目標輪帶空 think,loss 只算目標輪)。最貼近推論,但資料量變多,需要自己做前處理,verl 現成的 MultiTurnSFTDataset 不會這樣切。
- 所有 assistant 輪都帶空 think、全部算 loss:序列內一致、實作簡單,代價是歷史輪的格式和推論時(被剝掉)有落差。不少框架的 non-thinking 訓練是這樣做的,實務上多半能接受,但我不確定每個框架的細節,別當成標準。

(3) 不確定 verl sft 實作
verl 可能是逐輪單獨套 template,結果是所有 assistant 輪都沒有 think,等於第 3 種做法的「全部不帶」版本。如果還想沿用這個 dataset:

推論時改成不用 enable_thinking=False(讓 prompt 結尾不帶空 think),先實測微調後的模型會不會自己吐 <think>。
或者自己改 dataset,讓目標輪前面補上空 think 的 token,並且不把它算進 loss。

