#!/usr/bin/env bash
set -euo pipefail

# Nero 双臂 pickplace_new joint 异步推理 server。
# 先启动这个脚本，再在另一个终端启动 run_nero_pickplace_new_infer_client.sh。

cd /home/chenglong/workplace/nero_teleop_ws/lerobot

source /home/chenglong/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# 强制离线加载本地 checkpoint，避免 server 卡在 huggingface.co 网络请求。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

args=(
  # 监听所有网卡，client 可通过 127.0.0.1:8080 连接
  --host=0.0.0.0
  # gRPC server 端口
  --port=8080
  # server 侧环境频率，用于 RTC 延迟 step 换算
  --fps=30
  # 观测相似过滤阈值，越小越容易触发新推理
  --obs_similarity_atol=0.09
  # 保持当前 joint 推理设置；如果 checkpoint 无 rtc_config，server 会退化为普通 async 推理
  --async_rtc.enabled=true
  # RTC 延迟估计使用最近延迟的 p90，避免首包慢推理长期锁死 inference_delay
  --async_rtc.latency_quantile=0.9
  # 将 RTC 内部统计写入本次录制目录 rtc_debug.jsonl
  --async_rtc.debug_dump.enabled=true
)

python -m lerobot.async_inference.policy_server "${args[@]}"
