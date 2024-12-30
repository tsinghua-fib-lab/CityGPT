SRC=/your_path/chatglm3-6b
DST=/your_path/chatglm3-6B-v22.5-newyork-20240521/

FILES=("config.json" "tokenizer_config.json" "tokenizer.model" "tokenization_chatglm.py" "quantization.py" "configuration_chatglm.py" "modeling_chatglm.py" "special_tokens_map.json")

for file in ${FILES[@]}
    do
        echo $file
        cp $SRC/$file $DST/$file
    done
