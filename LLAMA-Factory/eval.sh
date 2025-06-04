export CUDA_VISIBLE_DEVICES=4

exec -a "llm-eval-mmlu@fengjie"  python src/evaluate.py \
    --model_name_or_path /data1/fengjie/init_ckpt/agentlm-7b \
    --template llama2 \
    --finetuning_type full \
    --task mmlu \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 4