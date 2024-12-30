import datasets
import pandas as pd
import uuid
import json
from datasets import Dataset
import shutil
from pathlib import Path
import jsonlines
import os
import json
import glob

from faker import Faker
fake = Faker()
Faker.seed(0)

from token_statis import token_statis
import warnings
warnings.filterwarnings('ignore')

USER_ROLE = "user"
ASSIS_ROLE = "assistant"

def message_gen(x_in, y_out):
    return [
        {"content":str(x_in), "role":USER_ROLE},
        {"content":str(y_out), "role":ASSIS_ROLE}
    ]

def record_id():
    return str(fake.uuid4()).split("-")[-1]


# define for sharegpt
def message_convert(info):
    new_info = []
    for i, x in enumerate(info):
        if x["from"]=="human":
            role = USER_ROLE
        else:
            role = ASSIS_ROLE
        new_info.append({"role":role, "content":str(x["value"])})

    return new_info

def message_convert_v2(info):
    new_info = []
    for i, x in enumerate(info):
        new_info.append({"role":USER_ROLE, "content":str(x["human"])})
        new_info.append({"role":ASSIS_ROLE, "content":str(x["assistant"])})

    return new_info

def is_human_first(x):
    if x[0]["from"]=="human":
        return True
    else:
        return False

def filter_conversation(x):
    
    for xx in x:
        info = xx["value"]
        if "你是谁？" in info: 
            # print(x)
            return True
        if "我是ChatGPT" in info:
            # print(x)
            return True
        if "I apologize" in info:
            # print(x)
            return True
        if "我是GPT" in info:
            # print(x)
            return True
        if "我是由OpenAI创造" in info:
            # print(x)
            return True
        if "I am GPT" in info:
            return True
    else:
        return False

def process_sharegpt(data_path=None, data_name=None):
    # https://huggingface.co/datasets/shibing624/sharegpt_gpt4
    data = pd.read_json(data_path, lines=True)

    data["is_human_first"] = data.apply(lambda x: is_human_first(x["conversations"]), axis=1)
    print("delete {} records because the first info is not from human".format(data[data["is_human_first"]==False].shape[0]))
    data = data[data["is_human_first"]==True]
    data["should_be_filtered"] = data.apply(lambda x: filter_conversation(x["conversations"]), axis=1)
    print("delete {} records because they contains other identity info".format(data[data["should_be_filtered"]==True].shape[0]))
    print(data[data["should_be_filtered"]==True].head(3))
    data = data[data["should_be_filtered"]==False]

    data["prompt_id"] = data.apply(lambda x: record_id(), axis=1)
    data["data_source"] = "sharegpt_zh_38k_format" if data_name is None else data_name
    data["prompt"] = data.apply(lambda x: x["conversations"][0]["value"], axis=1)
    data["messages"] = data.apply(lambda x: message_convert(x["conversations"]), axis=1)

    print("processing sharegpt_zh_38k_format with {} records".format(len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]


def process_sharegpt_en(data_path=None, data_name=None):
    # https://huggingface.co/datasets/shibing624/sharegpt_gpt4
    data = pd.read_json(data_path, lines=True)

    data["is_human_first"] = data.apply(lambda x: is_human_first(x["conversations"]), axis=1)
    print("delete {} records because the first info is not from human".format(data[data["is_human_first"]==False].shape[0]))
    data = data[data["is_human_first"]==True]
    data["should_be_filtered"] = data.apply(lambda x: filter_conversation(x["conversations"]), axis=1)
    print("delete {} records because they contains other identity info".format(data[data["should_be_filtered"]==True].shape[0]))
    print(data[data["should_be_filtered"]==True].head(3))
    data = data[data["should_be_filtered"]==False]

    data["prompt_id"] = data.apply(lambda x: record_id(), axis=1)
    data["data_source"] = "sharegpt_gpt4" if data_name is None else data_name
    data["prompt"] = data.apply(lambda x: x["conversations"][0]["value"], axis=1)
    data["messages"] = data.apply(lambda x: message_convert(x["conversations"]), axis=1)

    print("processing sharegpt4_en with {} records".format(len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]

def process_sharegpt_90k(data_path=None, data_name=None):
    # https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k
    data = pd.read_json(data_path, lines=True)

    data["prompt_id"] = data["conversation_id"]
    data["data_source"] = "sharegpt_90k" if data_name is None else data_name
    data["prompt"] = data.apply(lambda x: x["conversation"][0]["human"], axis=1)
    data["messages"] = data.apply(lambda x: message_convert_v2(x["conversation"]), axis=1)

    print("processing sharegpt4_en with {} records".format(len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]


def process_open_platypus(data_path=None, data_name=None):
    # https://huggingface.co/datasets/garage-bAInd/Open-Platypus
    data = pd.read_parquet(data_path)
    
    data.rename(columns={"input": "messages", "instruction": "prompt"}, inplace=True)
    data["prompt_id"] = data.apply(lambda x: record_id(), axis=1)
    data["messages"] = data.apply(lambda x: message_gen(x["prompt"], x["output"]), axis=1)
    data["data_source"] = data["data_source"]+ ("_OpenPlatypus" if data_name is None else data_name)
    
    print("processing OpenPlatypus with {} records".format(len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]

def process_alpaca_evol_instruct(data_path=None, data_name=None):
    # https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k
    data = pd.read_json(data_path)
    
    data.rename(columns={"instruction": "prompt"}, inplace=True)
    data["prompt_id"] = data.apply(lambda x: record_id(), axis=1)
    data["messages"] = data.apply(lambda x: message_gen(x["prompt"], x["output"]), axis=1)
    data["data_source"] = "evol_instruct_70k" if data_name is None else data_name
    
    print("processing {} with {} records".format(data_name, len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]

def process_alpaca_evol_instruct_v2(data_path=None, data_name=None):
    # https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k
    data = pd.read_json(data_path)
    
    data["prompt_id"] = data["idx"]
    data["prompt"] = data.apply(lambda x: x["conversations"][0]["value"], axis=1)
    data["messages"] = data.apply(lambda x: message_convert(x["conversations"]), axis=1)
    data["data_source"] = "WizardLM_evol_instruct_V2_196k" if data_name is None else data_name
    
    print("processing {} with {} records".format(data_name, len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]

def process_hc3_chinese(data_path=None, data_name=None):
    # https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese
    data = pd.read_json(data_path, lines=True)
    
    data.rename(columns={"question": "prompt"}, inplace=True)
    data["prompt_id"] = data.apply(lambda x: record_id(), axis=1)
    # 注：hc3-chinese每个回答是个只含有一个元素的列表
    data["messages"] = data.apply(lambda x: message_gen(x["prompt"], x["human_answers"][0]), axis=1)
    data["data_source"] = data["source"]+ ("_HC3-Chinese" if data_name is None else data_name)
    
    print("processing hc3_chinese with {} records".format(len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]

def process_alpaca_gpt4_data_zh(data_path=None, data_name=None):
    # https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/tree/main/data
    data = pd.read_json(data_path, lines=False)
    
    data.rename(columns={"instruction": "prompt"}, inplace=True)
    data["prompt_id"] = data.apply(lambda x: record_id(), axis=1)
    data["messages"] = data.apply(lambda x: message_gen(x["prompt"], x["output"]), axis=1)
    data["data_source"] = "alpaca_gpt4_data_zh" if data_name is None else data_name
    
    print("processing alpaca_gpt4_data_zh with {} records".format(len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]

def process_geosignal(data_path=None, data_name=None):
    # https://github.com/davendw49/k2/tree/main/data/geosignal
    data = pd.read_json(data_path, lines=False)
    data = data[data["type"]=="geo"]
    
    data.rename(columns={"instruction": "prompt"}, inplace=True)
    data["prompt_id"] = data.apply(lambda x: record_id(), axis=1)
    data["messages"] = data.apply(lambda x: message_gen(x["prompt"], x["output"]), axis=1)
    data["data_source"] = data["type"]+"_"+data["category"]+ ("_k2_geosignal" if data_name is None else data_name)
    
    print("processing geosignal with {} records".format(len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]

def process_transgpt(data_path=None, data_name=None):
    # https://huggingface.co/datasets/DUOMO-Lab/TransGPT-sft
    data = pd.read_json(data_path, lines=True)
    
    data["prompt"] = data.apply(lambda x: json.loads(x[0])["instruction"].strip("\n"), axis=1)
    data["output"] = data.apply(lambda x: json.loads(x[0])["output"].strip("\n"), axis=1)
    data["prompt_id"] = data.apply(lambda x: record_id(), axis=1)
    data["messages"] = data.apply(lambda x: message_gen(x["prompt"], x["output"]), axis=1)
    data["data_source"] = "transgpt" if data_name is None else data_name
    
    print("processing transgpt with {} records".format(len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]

def process_benchmark(data_path=None, data_name=None):
    data = pd.read_json(data_path, lines=True)

    data.rename(columns={"input": "prompt"}, inplace=True)
    data["prompt_id"] = data.apply(lambda x: record_id(), axis=1)
    data["messages"] = data.apply(lambda x: message_gen(x["prompt"], x["target"]), axis=1)
    data["data_source"] = data["type"]+"_"+ ("self-benchmark" if data_name is None else data_name)

    print("processing self benchmark with {} records".format(len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]

def get_inout_from_messages(messages, message_type="input"):
    if message_type=="input":
        if messages[0]["role"] == "system":
            prefix = messages[0]["content"]
        else:
            prefix = ""
        
        for info in messages:
            if info["role"]=="user":
                return prefix + "\n" + info["content"]
    elif message_type=="output":
        for info in messages:
            if info["role"]=="assistant":
                # TODO 临时
                if info["content"] is None:
                    print("I donot know the answer.")
                    return "I donot know the answer."
                else:
                    return info["content"]
    else:
        return "TEMPLATE FOR INCORRECT INFO"

def process_ugi_data(data_path=None, data_name=None, data_path_list=None):
    data_concat = []
    if data_path_list != None and data_path==None:
        for data_path in data_path_list:
            data = pd.read_json(data_path, lines=True)
            data_concat.append(data)
        data = pd.concat(data_concat, axis=0)
    else:
        data = pd.read_json(data_path, lines=True)    

    data["messages"] = data["diag"]
    data["prompt"] = data.apply(lambda x: get_inout_from_messages(x["messages"], "input"), axis=1)
    data["output"] = data.apply(lambda x: get_inout_from_messages(x["messages"], "output"), axis=1)
    data["prompt_id"] = data["id"]
    data["data_source"] = data_name+"_"+data["task"]

    print("processing {} with {} records".format(data_name, len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]

def process_LIMA(data_path=None, data_name=None):
    data = pd.read_json(data_path, lines=True)
    data["prompt"] = data.apply(lambda x: x["conversations"][0], axis=1)
    data["output"] = data.apply(lambda x: x["conversations"][1], axis=1)
    data["prompt_id"] = data.apply(lambda x: record_id(), axis=1)
    data["messages"] = data.apply(lambda x: message_gen(x["prompt"], x["output"]), axis=1)
    data["data_source"] = data_name + "_" + data["source"]

    print("processing {} Train dataset with {} records".format(data_name, len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]


def process_agentinstruct(data_path=None, data_name=None):
    import os
    datas = []
    for file in os.listdir(data_path):
        data_c = pd.read_parquet(os.path.join(data_path, file))
        datas.append(data_c)
    data = pd.concat(datas)

    data["prompt_id"] = data["id"]
    data["data_source"] = "AgentInstruct" if data_name is None else data_name
    data["prompt"] = data.apply(lambda x: x["conversations"][0]["value"], axis=1)
    data["messages"] = data.apply(lambda x: message_convert(x["conversations"]), axis=1)

    print("processing {} dataset with {} records".format(data_name, len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]


def process_utralchat_200k(data_path=None, data_name=None):
    import os
    datas = []
    for file in os.listdir(data_path):
        if "sft" not in file:
            continue
        data_c = pd.read_parquet(os.path.join(data_path, file))
        datas.append(data_c)
    data = pd.concat(datas)

    data["data_source"] = "utralchat_200k" if data_name is None else data_name
    print("processing {} dataset with {} records".format(data_name, len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]


def get_question(info):
    INIT_PROMPT = "The following is a multiple-choice question about location of POI. Please choose the most suitable one among A, B, C, D, E and F as the answer to this question. Please output the option directly.\n"
    
    example = INIT_PROMPT + 'Question: ' + info["question"]
    for s in ["A", "B", "C", "D", "E", "F"]:
        if s in info.keys():
            example += f'\n{s}. {info[f"{s}"]}'
    example += '\nAnswer:'

    return example

def extract_answer(info):
    return info["answer"]+". "+str(info[info["answer"]])

def process_eval_train(data_path=None, data_name=None, max_validation=200):
    data = pd.read_csv(data_path, header="infer")

    if data.shape[0]<=1:
        return
    
    data = data.head(max_validation)

    data["prompt_id"] = data.apply(lambda x: record_id(), axis=1)
    data["data_source"] = "eval2train" if data_name is None else data_name
    data["prompt"] = data.apply(lambda x: get_question(x), axis=1)
    data["output"] = data.apply(lambda x: extract_answer(x), axis=1)
    data["messages"] = data.apply(lambda x: message_gen(x["prompt"], x["output"]), axis=1)

    return data[["prompt", "messages", "prompt_id", "data_source"]]


def main_factory(mode="sharegpt", token_path="", save_path=None, common_path=None, citywalk_name=None, data_mix=None, cityeval_name=None, cityaddr_name=None, spatialreasoning_name=None, self_reasoning_name=None):
    print("start processing")

    # TODO 新数据请依次往后添加，以保证前序数据的uuid保持不变
    general_path = os.path.join(common_path, "general")
    data_collection = {}

    if "wizardllm_143k" in data_mix:
        wizardllm_143k = process_alpaca_evol_instruct_v2(
            data_path=os.path.join(general_path, "WizardLM_evol_instruct_V2_196k/WizardLM_evol_instruct_V2_143k.json"), 
            data_name="WizardLM_evol_instruct_V2_143k"
        )
        token_statis(wizardllm_143k["messages"].to_list(), token_path=token_path, data_mode="sharegpt")
        data_collection["wizardllm_143k"] = wizardllm_143k
    if "platypus_25k" in data_mix:
        platypus_25k = process_open_platypus(
            data_path=os.path.join(general_path, "OpenPlatypus.parquet"),
            data_name="Open-Platypus-25K"
        )
        token_statis(platypus_25k["messages"].to_list(), token_path=token_path, data_mode="sharegpt")
        data_collection["platypus_25k"]=platypus_25k

    if "LIMA" in data_mix:
        LIMA_data = process_LIMA(
            data_path=os.path.join(general_path, "LIMA-train.jsonl"),
            data_name="LIMA-1.3k"
        )
        token_statis(LIMA_data["messages"].to_list(), token_path=token_path, data_mode="sharegpt")
        data_collection["LIMA"] = LIMA_data

    if "utralchat_200k" in data_mix:
        utralchat_200k = process_utralchat_200k(
            data_path=os.path.join(general_path, "utralchat_200k/data"),
            data_name="utralchat_200k_sft"
        )
        token_statis(utralchat_200k["messages"].to_list(), token_path=token_path, data_mode="sharegpt")
        data_collection["utralchat_200k"] = utralchat_200k

    if "sharegpt_90k_en" in data_mix:
        sharegpt_90k_en = process_sharegpt_90k(
            data_path=os.path.join(general_path, "ShareGPT-Chinese-English-90k/sharegpt_jsonl/common_en_70k.jsonl"),
            data_name="sharegpt_90k_en"
        )
        token_statis(sharegpt_90k_en["messages"].to_list(), token_path=token_path, data_mode="sharegpt")
        data_collection["sharegpt_90k_en"] = sharegpt_90k_en

    if "sharegpt_90k_zh" in data_mix:
        sharegpt_90k_zh = process_sharegpt_90k(
            data_path=os.path.join(general_path, "ShareGPT-Chinese-English-90k/sharegpt_jsonl/common_zh_70k.jsonl"),
            data_name="sharegpt_90k_zh"
        )
        token_statis(sharegpt_90k_zh["messages"].to_list(), token_path=token_path, data_mode="sharegpt")
        data_collection["sharegpt_90k_zh"] = sharegpt_90k_zh

    if "agent_instruct" in data_mix:
        agent_instruct = process_agentinstruct(
            data_path=os.path.join(general_path, "AgentInstruct"),
            data_name="AgentInstruct"
        )
        token_statis(agent_instruct["messages"].to_list(), token_path=token_path, data_mode="sharegpt")
        data_collection["agent_instruct"] = agent_instruct

    if "eval2train" in data_mix:
        data_collect = []
        # TODO 该数据地址需要进一步修改
        for file in glob.glob(os.path.join(common_path, "eval", cityeval_name)):
            data = process_eval_train(file, file.split("/")[-1], max_validation=200)
            data_collect.append(data)
        eval2train = pd.concat(data_collect)
        token_statis(eval2train["messages"].to_list(), token_path=token_path, data_mode="sharegpt")
        data_collection["eval2train"] = eval2train

    # 基于citywalk生成的城市空间环境数据集
    city_world_model_data_path = os.path.join(common_path, "citywalk")
    if "citywalk" in data_mix:
        cwm_citywalk_data = process_ugi_data(None, data_name="cwm_citywalk_data", data_path_list=[os.path.join(city_world_model_data_path, citywalk_name)])
        cwm_citywalk_data = cwm_citywalk_data.sample(frac=1)
        token_statis(cwm_citywalk_data["messages"].to_list(), token_path=token_path, data_mode="sharegpt")
        data_collection["citywalk"] = cwm_citywalk_data

    # 丰富输入数据样式
    if "cityaddr" in data_mix:
        cwm_cityaddr_data = process_ugi_data(None, data_name="cwm_cityaddr_data", data_path_list=[os.path.join(city_world_model_data_path, cityaddr_name)])
        cwm_cityaddr_data = cwm_cityaddr_data.sample(frac=1)
        token_statis(cwm_cityaddr_data["messages"].to_list(), token_path=token_path, data_mode="sharegpt")
        data_collection["cityaddr"] = cwm_cityaddr_data

    # 外部空间推理数据
    if "spatialreasoning" in data_mix:
        cwm_spatial_data = process_ugi_data(None, data_name="spatial-reasoning", data_path_list=[os.path.join(city_world_model_data_path, spatialreasoning_name)])
        cwm_spatial_data = cwm_spatial_data.sample(frac=1)
        token_statis(cwm_spatial_data["messages"].to_list(), token_path=token_path, data_mode="sharegpt")
        data_collection["spatialreasoning"] = cwm_spatial_data
    
    # 自行构建的空间推理数据
    city_world_model_data_path = os.path.join(common_path, "reasoning")
    if "self-reasoning" in data_mix:
        data_path_list = [os.path.join(city_world_model_data_path, file) for file in self_reasoning_name.split(",")]
        print(data_path_list)
        cwm_self_data = process_ugi_data(None, data_name="self-reasoning", data_path_list=data_path_list)
        cwm_self_data = cwm_self_data.sample(frac=1)
        token_statis(cwm_self_data["messages"].to_list(), token_path=token_path, data_mode="sharegpt")
        data_collection["self-reasoning"] = cwm_self_data

    print("begin merge datasets")
    # 混合比例可以独立实验
    data_list = []
    for key in data_mix:
        if data_mix[key]>0:
            data_list.append(data_collection[key].head(data_mix[key]))
        else:
            data_list.append(data_collection[key])

    df = pd.concat(data_list)
    df = df.sample(frac = 1)
 
    messages = df["messages"].to_list()
    print("overall data")
    token_statis(messages, token_path=token_path, data_mode="sharegpt")

    if mode == "sharegpt":
        data_new = []
        for item in messages:
            info = {"conversations":[]}
            for it in item:
                if it["role"]=="user":
                    role_from = "human"
                elif it["role"]=="assistant":
                    role_from = "gpt"
                info["conversations"].append({"from": role_from, "value": it["content"]})
            data_new.append(info)
    elif mode == "alpaca":
        data_new = []
        for item in messages:
            assert item[0]["role"]=="user", "第一个需要来自用户"
            assert item[1]["role"]=="assistant", "第二个来自gpt"
            info = {
                "instruction": item[0]["content"],
                "output": item[1]["content"],
                "system": "",
                "history":[]
                }
            for i in range(0, int(len(item)/2)*2, 2):
                try:
                    info["history"].append([item[i]["content"], item[i+1]["content"]])
                except IndexError as e:
                    print(i, e)
            data_new.append(info)
    else:
        raise NotImplementedError
    
    with open(save_path, "w") as wid:
        json.dump(data_new, wid, indent=4, ensure_ascii=False)
    
    with open(save_path, "r") as fid:
        test_data = json.load(fid)

def get_best_match(target, options):
    # 使用 difflib.get_close_matches 找出最接近的匹配
    import difflib
    matches = difflib.get_close_matches(target, options, n=1, cutoff=0.0)
    if matches:
        return matches[0]
    else:
        return None


if __name__ == "__main__":

    INPUT_CITYWALK_DATA="citywalk-paris-mock-v11.1-eng-eng.jsonl"
    INPUT_ADDRESS_DATA="address-paris-v11.3-eng-eng-selected.jsonl"
    INPUT_SPATIAL_DATA="additional-wudaokou_small-mock-v10.2.jsonl"
    INPUT_SELF_DATA="direction_reasoning_multistyle_v3.jsonl,distance_reasoning_multistyle_v3.jsonl,citywalk-wudaokou_small-mock-v10.2-chinese-v3.jsonl,citywalk-paris-mock-v11.1-eng-eng-v3.jsonl"
    INPUT_CITYEVAL_DATA="wangjing/v1/*"
    # 输出数据的数据集名称，将在后续训练脚本中使用
    OUTPUT_DATA_NAME_KEY="citygpt-citywalk-v11-cwm-v22.4-paris-multistyle-v3-wudaokou"


    # 通用数据混合参数控制
    DATA_MIX = {
        "platypus_25k": 20000,
        "utralchat_200k": 10000,
        # "sharegpt_90k_zh": 1000,
        "sharegpt_90k_en": 10000,
        "agent_instruct": -1,
        "eval2train": -1,
        "citywalk":-1,
        "cityaddr":-1,
        "spatialreasoning":-1,
        "self-reasoning":-1
    }

    mode = "alpaca"
    data_name = "{}-{}.json".format(OUTPUT_DATA_NAME_KEY, mode)
    save_path = os.path.join("../data/", data_name)

    token_path = "/your_path/Qwen1___5-7B-Chat/"
    common_path = "/your_path/city_world_model" # 下面有citywalk，merge，general，cache四个路径
    
    print("data_version:{}".format(OUTPUT_DATA_NAME_KEY))
    main_factory(mode=mode, token_path=token_path, save_path=save_path, common_path=common_path, citywalk_name=INPUT_CITYWALK_DATA, data_mix=DATA_MIX, cityeval_name=INPUT_CITYEVAL_DATA, cityaddr_name=INPUT_ADDRESS_DATA, spatialreasoning_name=INPUT_SPATIAL_DATA, self_reasoning_name=INPUT_SELF_DATA)
    
    # 记录数据信息到dataset_info
    with open("../data/dataset_info.json") as fid:
        dataset_info = json.load(fid)
    dataset_info[OUTPUT_DATA_NAME_KEY] = {
        "file_name": data_name,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "response": "output",
            "system": "system",
            "history": "history"
            }
    }
    with open("../data/dataset_info.json", "w") as wid:
        json.dump(dataset_info, wid, indent=2, ensure_ascii=False)
    
    # 复制一份代码和数据到common_path进行存储备份
    os.popen("cp {} {}".format("../data/dataset_info.json", os.path.join(common_path, "merge")))
    os.popen("cp {} {}".format(save_path, os.path.join(common_path, "merge")))
    os.popen("cp {} {}".format("./data_process.py", os.path.join(common_path, "merge", "data_process_{}.py".format(OUTPUT_DATA_NAME_KEY))))
