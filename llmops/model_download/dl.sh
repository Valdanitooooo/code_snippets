#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

until /data/llmops/anaconda3/envs/llm/bin/huggingface-cli download --resume-download unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit --local-dir Qwen2.5-7B-Instruct-unsloth-bnb-4bit
do
    echo "下载过程中出现错误，1秒后重试..."
    sleep 1
done
echo "下载完成！"
