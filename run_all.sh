#!/bin/bash
# 依次训练全部模型（Mac GPU 上约 40 分钟）
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
$PY train.py --model laya --loss ce --steps 6000
$PY train.py --model laya --loss brier --steps 6000
$PY train.py --model nanojev --loss ce
$PY train.py --model gpt
$PY train.py --model laya_bert --loss ce
$PY train.py --model nanojev_qwen --loss ce
echo "ALL DONE"
