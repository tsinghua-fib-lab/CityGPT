PYTHON_ENV_SERVING="training2"
export CUDA_VISIBLE_DEVICES=3
source /usr/local/anaconda3/bin/activate $PYTHON_ENV_SERVING
MODEL_PATH="/data1/citygpt/init_ckpt/Qwen/Qwen2___5-7B-Instruct"
DATASET="citygpt-Paris-v24.6-SF-v2.3-part1"
SAVE_FILE="/data4/liutianhui/LLAMA-Factory/data/citygpt-qwen2.5-Paris-v24.6-SF-v2.3-ppl-part1.json"
DATASET_DIR="/data4/liutianhui/LLAMA-Factory/data"

python cal_ppl.py --model_name_or_path $MODEL_PATH --dataset $DATASET --save_name $SAVE_FILE --dataset_dir $DATASET_DIR