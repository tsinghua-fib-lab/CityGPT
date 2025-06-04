import json
import numpy as np
from multiprocessing import Pool
from transformers import AutoTokenizer


def token_counting(input):
    data = input["data"]
    token_path = input["token"]
    mode = input["mode"]

    # 增加truct_remote_code=True，方便加载一些没有被transformers直接支持的库
    tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)

    token_count = []
    for item in data:
        if mode == "sharegpt":
            messages = item
        elif mode == "alpaca":
            messages = []
            for i, his in enumerate(item["history"]):
                if i%2==0:
                    messages.append({"role": "user", "content": str(his)})
                else:
                    messages.append({"role": "assistant", "content": str(his)})
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(e)
            continue
        model_inputs = tokenizer([text], return_tensors="pt")
        token_count.append(model_inputs["input_ids"].size()[-1])
    
    return token_count


def token_statis(data, token_path, data_mode):
    data_pool = []
    for i in range(0, len(data), 1000):
        data_pool.append({"data": data[i:min(i+1000, len(data))], "token":token_path, "mode":data_mode})
    
    pool = Pool(64)
    result_list = pool.map(token_counting, data_pool)
    pool.close()
    pool.join()

    final_tokens = []
    for res in result_list:
        final_tokens.extend(res)

    # for debug
    # final_tokens = [token_counting({"data": data, "token":token_path, "mode":data_mode})]

    print("data samples:{} token counting:{:.4f}B".format(len(final_tokens), np.sum(final_tokens)/1e9))
    print("token distribution p50:{} p75:{} p90:{} p95:{}".format(
        np.percentile(final_tokens, 50, axis=0),
        np.percentile(final_tokens, 75, axis=0),
        np.percentile(final_tokens, 90, axis=0),
        np.percentile(final_tokens, 95, axis=0)
    ))

if __name__ == "__main__":
    TOKEN_PATH = "/data1/citygpt/init_ckpt/ZhipuAI/chatglm3-6b"
    DATA_PATH = "/data1/citygpt/datasets/city_world_model/merge/citygpt-citywalk-v8-cwm-v18.1-address-alpaca.json"

    with open(DATA_PATH) as fid:
        data = json.load(fid)
    token_statis(data=data, token_path=TOKEN_PATH, data_mode="alpaca")
