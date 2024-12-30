export CUDA_VISIBLE_DEVICES="0,3,4,5"
python run.py --datasets mmlu_gen gsm8k_gen bbh_gen \
--hf-path /data1/citygpt/model_zoo/citygpt-8b-20240523 \
--model-kwargs device_map='auto' trust_remote_code=True \
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
--max-seq-len 2048 \
--max-out-len 1000 \
--batch-size 4  \
--num-gpus 1

