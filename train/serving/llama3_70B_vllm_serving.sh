source /usr/local/anaconda3/bin/activate vllm
TP=1

if [ "$TP" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=5
  MAX_LEN=4096
  echo "running on sinlge GPU"
else
  ray stop
  sleep 5
  export CUDA_VISIBLE_DEVICES=5,6
  MAX_LEN=8192
  ray start --head --num-cpus=32 # 32 is required
  sleep 5
  ray status
  echo "ray restart first"
fi

exec -a "vllm-llama3-70B-awq" python -m vllm.entrypoints.openai.api_server \
  --served-model-name LLama3-70B-AWQ-4bit \
  --api-key token-fiblab-20240425 \
  --model model_path \
  --trust-remote-code \
  --host your_ip \
  --port your_port \
  --quantization awq \
  --swap-space 16 \
  --max-model-len $MAX_LEN \
  --disable-log-stats \
  --tensor-parallel-size $TP \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9

# vllm engine parameters: https://docs.vllm.ai/en/latest/models/engine_args.html
# vllm openai server parameters: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html