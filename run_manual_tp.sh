#!/bin/bash
# Launch manual TP server with proper NCCL collectives

cd /home/azureuser/multi-agent-test
source venv/bin/activate

# Kill existing
pkill -9 -f "manual_tp_server\|synthesis_server\|deepspeed" || true
sleep 2

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ENABLE_PROFILING=true
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Launch 4 processes with torchrun
torchrun --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29500 \
    serving/a100_cluster/manual_tp_server.py \
    --model-name meta-llama/Llama-2-70b-chat-hf \
    --port 8020 \
    --profiling-enabled



