#!/usr/bin/env bash
# 範本:GRPO + SGLang multi-turn + LLM-as-judge。
# verl 各版本的 config key 有差異(例如 multi_turn.enable / mode / agent.default_agent_loop),
# 若 hydra 報 unknown key,請對照你安裝版本的 docs 的 "Multi-turn Rollout Support" 與 config 說明調整。
set -euo pipefail

#ROOT="$(cd "$(dirname "$0")/.." && pwd)"
#ROOT="/mnt/yuzhe/projects/verl/yzk/multi_turn_agent"
ROOT="/mnt/yuzhe/projects/verl/yzk/poketomo"
export PYTHONPATH="$ROOT/tools:${PYTHONPATH:-}"        # 讓 tool yaml 的 class_name 能被 import

# ---- Judge(獨立服務,OpenAI 相容端點;請勿與訓練 GPU 搶資源)----
#export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://judge-host:8001/v1}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://service.ai-infra.aicoe.efoxconn.com/litellm/v1}"
#export JUDGE_MODEL="${JUDGE_MODEL:-your-judge-model}"
#export JUDGE_MODEL="${JUDGE_MODEL:-qwen3.8-27b}"
export JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-pro-b}"
#export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-32}"
export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"
export JUDGE_LOG_PATH="${JUDGE_LOG_PATH:-$ROOT/judge_log.jsonl}"   # 訓練中請定期抽查,防 reward hacking

#MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"            # 換成你的對話 SLM
#MODEL_PATH="${MODEL_PATH:-/mnt/yuzhe/models/Qwen3.5-2B-Base}"            # 換成你的對話 SLM
MODEL_PATH="${MODEL_PATH:-/mnt/yuzhe/models/Qwen3-0.6B-Base}"            # 換成你的對話 SLM
#DATA_DIR="${DATA_DIR:-$ROOT/data/parquet/jimmy}"
DATA_DIR="${DATA_DIR:-$ROOT/data/parquet}"
 
# =================== wandb ===================
project_name=companion123
experiment_name=qwen3-0.6b
default_local_dir=$ROOT/checkpoint/$experiment_name


#  reward_model.reward_manager=batch \
#  custom_reward_function.path="$ROOT/reward/companion_judge_reward.py" \
#  custom_reward_function.name=compute_score \
#  actor_rollout_ref.rollout.n=8 \
#  2B 得試試看用 lora => actor_rollout_ref.model.lora_rank、lora_alpha、target_modules
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/val.parquet" \
  data.return_raw_chat=True \
  data.train_batch_size=4 \
  data.max_prompt_length=4096 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$ROOT/config/companion_tools.yaml" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  reward.reward_manager.name=naive \
  reward.custom_reward_function.path="$ROOT/reward/companion_judge_reward.py" \
  reward.custom_reward_function.name=compute_score_single \
  trainer.logger='["console"]' \
  trainer.project_name=companion_rl \
  trainer.experiment_name=companion_grpo_judge \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=25 \
  trainer.total_epochs=5 "$@"


#grep -rn "custom_reward_function" verl/trainer/config/ | head
