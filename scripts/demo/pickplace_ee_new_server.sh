#!/usr/bin/env bash
set -euo pipefail

# Nero 双臂 pickplace EE-local-SE3 异步推理 server。
# 先启动这个脚本，再在另一个终端启动 pickplace_ee_new_client.sh。

cd /home/chenglong/workplace/nero_teleop_ws/lerobot

source /home/chenglong/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# 强制离线加载本地 checkpoint，避免 server 卡在 huggingface.co 网络请求。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

args=(
  # 监听所有网卡，client 可通过 127.0.0.1:8080 连接。
  --host=0.0.0.0
  # gRPC server 端口。
  --port=8080
  # server 侧环境频率，用于 action timing 和 RTC 延迟 step 换算。
  --fps=30
  # 观测相似过滤阈值；越小越容易触发新推理。
  --obs_similarity_atol=0.1
  # 保留 async RTC 开关；如果 checkpoint 没有 rtc_config，server 会自动降级为非 RTC。
  --async_rtc.enabled=true
  # RTC 延迟估计使用 rolling p90。
  --async_rtc.latency_quantile=0.9
  # 如果 RTC 实际启用，将 debug 数据写入本次 trace 目录。
  --async_rtc.debug_dump.enabled=true
)

python -m lerobot.async_inference.policy_server "${args[@]}"
