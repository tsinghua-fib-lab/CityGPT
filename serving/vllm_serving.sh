source /usr/local/anaconda3/bin/activate vllm
export CUDA_VISIBLE_DEVICES=4

MODEL_NAME=llama3-8B-local
MODEL_PATH=/data4/citygpt/init_ckpt/llama3/Meta-Llama-3-8B-Instruct/
IP_ADDRESS=""
PORT=""
API_KEY=""

exec -a "vllm-$MODEL_NAME@fengjie" python -m vllm.entrypoints.openai.api_server \
  --served-model-name $MODEL_NAME \
  --api-key $API_KEY \
  --model $MODEL_PATH \
  --trust-remote-code \
  --host $IP_ADDRESS \
  --port $PORT \
  --max-model-len 4096 \
  --disable-log-stats \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95

# autoAWQ https://docs.vllm.ai/en/latest/quantization/auto_awq.html
# vllm engine parameters: https://docs.vllm.ai/en/latest/models/engine_args.html
# vllm openai server parameters: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html