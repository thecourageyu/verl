from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

port = 7899
model_path = "qwen/qwen2.5-0.5b-instruct"
model_path = "/mnt/yuzhe/projects/verl/checkpoint/qwen2.5-3b/global_step_500/actor/huggingface"
model_path = "./qwen2.5-3b-500"

 # --tool-call-parser qwen25 
server_process, port = launch_server_cmd(
    f"python3 -m sglang.launch_server --model-path {model_path} --served-model-name qwen2.5-3b-500 --tool-call-parser qwen25 --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")

# 一定要寫到檔案內, 避免找不到 port
# python sglang_server.py > check_port.log

# merge and generate the model.safetensors
# python -m verl.model_merger  merge --backend fsdp  --local_dir /mnt/yuzhe/projects/verl/checkpoint/q
# wen2.5-3b/global_step_500/actor