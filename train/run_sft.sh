#### 需要关注并修改的关键参数，单独调用示例
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export ACCELERATE_LOG_LEVEL=info
# export FORCE_TORCHRUN=1 

llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
