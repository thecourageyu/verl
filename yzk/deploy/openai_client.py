import openai

port = 37335
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

# model="qwen/qwen2.5-0.5b-instruct",
response = client.chat.completions.create(
    model="qwen2.5-3b-500",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(f"Response: {response}")



resp = client.chat.completions.create(
    model="qwen2.5-3b-500",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
        }
    }],
    tool_choice="required",  # force a call, so you can isolate parsing from "did it decide to call"
)
print(resp, " <<< done!")
print(resp.choices[0].message.tool_calls)   # should be a populated list, not None/[]
print(resp.choices[0].message.content)

