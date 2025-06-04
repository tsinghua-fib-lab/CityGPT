export CUDA_VISIBLE_DEVICES="0,3,4,5"
python run.py --datasets mmlu_gen gsm8k_gen bbh_gen \
--hf-path /data4/citygpt/model_zoo/citygpt-llama3-London-v24.6-SF-v2.3-filtered_10_percent-loss-un-pk-20250406/ \
--model-kwargs device_map='auto' trust_remote_code=True \
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
--max-seq-len 2048 \
--max-out-len 1000 \
--batch-size 4  \
--num-gpus 1

