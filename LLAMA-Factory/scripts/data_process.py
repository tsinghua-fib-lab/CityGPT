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
    # str(x)保证x被解析为string而不是其他奇怪的东西，比如含有代码等东西，小心使用，可能被导致一些错误
    return [
        {"content":str(x_in), "role":USER_ROLE},
        {"content":str(y_out), "role":ASSIS_ROLE}
    ]

def record_id():
    return str(fake.uuid4()).split("-")[-1]


def process_self(data_path=None, data_name=None):
    # 自行构造的身份认知数据
    data = pd.read_csv(data_path, header="infer")
    
    data["prompt_id"] = data.apply(lambda x: record_id(), axis=1)
    data["data_source"] = "FIB-Lab" if data_name is None else data_name
    data["prompt"] = data["input"]
    data["messages"] = data.apply(lambda x: message_gen(x["input"], x["output"]), axis=1)
    
    print("processing self data with {} records".format(len(data)))
    return data[["prompt", "messages", "prompt_id", "data_source"]]

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
    # 天然多轮数据
    # https://huggingface.co/datasets/shibing624/sharegpt_gpt4
    data = pd.read_json(data_path, lines=True)

    # 过滤条件还可以继续完善，比如要求数据必须为user-assistant-user-assistant
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
    # 天然多轮数据
    # https://huggingface.co/datasets/shibing624/sharegpt_gpt4
    data = pd.read_json(data_path, lines=True)

    # 过滤条件还可以继续完善，比如要求数据必须为user-assistant-user-assistant
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
    # 天然多轮数据
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
        
        # TODO 只是逻辑上随意取一个，对于多轮支持存在问题
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


def main():
    print("start processing")
    # 新数据请依次往后添加，以保证前序数据的uuid保持不变
    trans_data = process_transgpt(data_path="/data1/fengjie/datasets/TransGPT/TransGPT-sft.json")
    geo_data = process_geosignal(data_path="/data1/fengjie/datasets/k2/geosignal.json")
    alpaca_data = process_alpaca_gpt4_data_zh(data_path="/data1/fengjie/datasets/alpaca_gpt4_data_zh.json")
    hc3_data = process_hc3_chinese(data_path="/data1/fengjie/datasets/HC3-Chinese-all.jsonl")
    evol_data = process_alpaca_evol_instruct(data_path="/data1/fengjie/datasets/alpaca_evol_instruct_70k.json")
    platypus_data = process_open_platypus(data_path="/data1/fengjie/datasets/OpenPlatypus.parquet")
    sharegpt_data = process_sharegpt("/data1/fengjie/datasets/sharegpt_zh_38K_format.jsonl")
    self_data = process_self("/data1/fengjie/datasets/citygpt_self_v2.csv")
    # 自我身份认知数据数量有限，从数量上进行提前扩充，后续增加更多自我身份认知数据后不再扩充
    self_data_extend = pd.concat([self_data]*10, ignore_index=True)

    #增加评测数据，进行极限测试
    self_benchmark = process_benchmark(data_path="/data1/fengjie/alignment-handbook/evaluation/benchmark/benchmark-v1.0.jsonl")
    print(self_benchmark.head(1))
    self_benchmark_extend = pd.concat([self_benchmark]*10, ignore_index=True)

    print("begin merge datasets")
    df = pd.concat([
        self_data_extend,
        self_benchmark_extend,
        sharegpt_data.head(10000),
        platypus_data.head(10000),
        evol_data.head(10000),
        hc3_data.head(3000),
        alpaca_data.head(3000),
        geo_data.head(4000),
        trans_data.head(2000)])
    df = df.sample(frac = 1)
    print(df.head(5))

    df2 = pd.concat([
        self_data,
        sharegpt_data.head(100),
        platypus_data.tail(100), 
        evol_data.tail(100),
        hc3_data.tail(50),
        alpaca_data.tail(100),
        geo_data.tail(100),
        trans_data.tail(50)])
    df2 = df2.sample(frac = 1)

    # 增加system prompt，已统一身份认知，比较简单，牺牲灵活性，临时处理
    # system_prompt = "你是城市大模型CityGPT，由清华大学电子系FIBLab实验室开发。你对城市生活及其相关领域研究比较了解，请尽量准确、严谨的回答用户问题。你的知识截止到2023年11月。"
    # df["messages"] = df["messages"].apply(lambda x: [{"content":system_prompt,"role":"system"}] + x )
    # df2["messages"] = df2["messages"].apply(lambda x: [{"content":system_prompt,"role":"system"}] + x )

    dataset1 = Dataset.from_pandas(df)
    dataset2 = Dataset.from_pandas(df2)

    save_path = "/data1/fengjie/datasets/city_sft/v6-removesys/"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    print("begin save dataset, training samples:{} testing samples:{}".format(df.shape, df2.shape))
    shutil.copyfile(
        "/data1/fengjie/alignment-handbook/scripts/data_preprocess.py", 
        save_path + "data_preprocess.py")
    dataset2.to_parquet(save_path + "test_city_sft.parquet")
    dataset1.to_parquet(save_path + "train_city_sft.parquet")

    # df.to_json("/data1/fengjie/datasets/city_sft.jsonl", orient="records", lines=True, force_ascii=False)


def main_ugi():
    print("start processing")
    # 新数据请依次往后添加，以保证前序数据的uuid保持不变
    evol_data = process_alpaca_evol_instruct(data_path="/data1/fengjie/datasets/alpaca_evol_instruct_70k.json")
    platypus_data = process_open_platypus(data_path="/data1/fengjie/datasets/OpenPlatypus.parquet")
    LIMA_data = process_LIMA("/data1/fengjie/datasets/LIMA-train.jsonl", "LIMA")
    share_gpt = process_sharegpt_en("/data1/fengjie/datasets/sharegpt_gpt4.jsonl", "sharegpt4_en")
    ugi_agent_data = process_ugi_data(None, data_name="ugi", data_path_list=[
        # "/data1/fengjie/datasets/ugi_agent/Task1/records_mobility_anchorGen.jsonl",
        # "/data1/fengjie/datasets/ugi_agent/Task1/records_mobility_planGen.jsonl",
        # "/data1/fengjie/datasets/ugi_agent/Task2/citysearch_task1.jsonl",
        # "/data1/fengjie/datasets/ugi_agent/Task2/citysearch_task2.jsonl",
        # "/data1/fengjie/datasets/ugi_agent/Task2/citysearch_task3.jsonl",
        # "/data1/fengjie/datasets/ugi_agent/Task2/citysearch_task4.jsonl",
        # "/data1/fengjie/datasets/ugi_agent/Task3/Indicator_Prediction.jsonl",
        # "/data1/fengjie/datasets/ugi_agent/Task3/OD_Prediction.jsonl",
        # "/data1/fengjie/datasets/ugi_agent/Task3/Site_Selection.jsonl",
        # "/data1/fengjie/datasets/ugi_agent/Task4/records_final_train_1.17.jsonl",
        # "/data1/fengjie/datasets/ugi_agent/Task4/records_final_train_1.28.jsonl",
        # "/data1/fengjie/datasets/ugi_agent/Task4/records_final_val_cot_2-shot_4.10-v1-remove-examples.jsonl"
        "/data1/fengjie/datasets/ugi_agent/Task4/finetune_data_20240519.json"
        # "/data1/fengjie/datasets/ugi_agent/WorldModel/wudaokou-gpt4-train-all.jsonl"
    ])

    print("begin merge datasets")
    # 混合比例可以独立实验
    df = pd.concat([
        platypus_data.head(10000),
        evol_data.head(3000),
        LIMA_data,
        share_gpt.head(3000),
        ugi_agent_data])
    df = df.sample(frac = 1)
    print(df.head(5))
    print(ugi_agent_data.tail(5))

    df2 = pd.concat([
        platypus_data.tail(100), 
        evol_data.tail(100),])
    df2 = df2.sample(frac = 1)

    dataset1 = Dataset.from_pandas(df)
    dataset2 = Dataset.from_pandas(df2)

    # alighnment-handbook 数据
    save_path = "/data1/fengjie/datasets/city_sft/ugi-intent-v9/"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    print("begin save dataset, training samples:{} testing samples:{}".format(df.shape, df2.shape))
    shutil.copyfile(
        "/data1/fengjie/alignment-handbook/scripts/data_preprocess.py", 
        save_path + "data_preprocess.py")
    dataset2.to_parquet(save_path + "test_city_ugi.parquet")
    dataset1.to_parquet(save_path + "train_city_ugi.parquet")

    # llama-factory 格式数据
    save_path = "../data/citygpt-ugi-intent-v9-alpaca.json"
    messages = df["messages"].to_list()
    data_new = []
    for item in messages:
        if item[0]["role"] == "system":
            system_info = item[0]["content"]
            assert item[1]["role"]=="user", "第一个需要来自用户"
            assert item[2]["role"]=="assistant", "第二个来自gpt"
            user_info = item[1]["content"]
            ass_info = item[2]["content"]
        else:
            assert item[0]["role"]=="user", "第一个需要来自用户"
            assert item[1]["role"]=="assistant", "第二个来自gpt"
            user_info = item[0]["content"]
            ass_info = item[1]["content"]
            system_info = ""

        info = {
            "instruction": user_info,
            "output": ass_info,
            "system": system_info,
            "history":[]
            }
        for i in range(0, int(len(item)/2)*2, 2):
            try:
                info["history"].append([item[i]["content"], item[i+1]["content"]])
            except IndexError as e:
                print(i, e)
        data_new.append(info)
    
    with open(save_path, "w") as wid:
        json.dump(data_new, wid, indent=4, ensure_ascii=False)
    
    with open(save_path, "r") as fid:
        test_data = json.load(fid)


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
        # TODO 格式数据参与多次训练，确认是否可以帮助模型,，注意是否需要保留和更新！！！
        data_collection["eval2train"] = eval2train # pd.concat([eval2train] * 5, ignore_index=True) #eval2train
    
    # 自我身份认知数据数量有限，从数量上进行提前扩充，后续增加更多自我身份认知数据后不再扩充
    # self_data = process_self("/data1/fengjie/datasets/citygpt_self_v2.csv")
    # self_data_extend = pd.concat([self_data]*10, ignore_index=True)

    # 面向UGI任务的agent训练数据
    # ugi_agent_data = process_ugi_data(None, data_name="ugi", data_path_list=[
    #     # "/data1/fengjie/datasets/ugi_agent/Task1/records_mobility_anchorGen.jsonl",
    #     # "/data1/fengjie/datasets/ugi_agent/Task1/records_mobility_planGen.jsonl",
    #     # "/data1/fengjie/datasets/ugi_agent/Task2/citysearch_task1.jsonl",
    #     # "/data1/fengjie/datasets/ugi_agent/Task2/citysearch_task2.jsonl",
    #     # "/data1/fengjie/datasets/ugi_agent/Task2/citysearch_task3.jsonl",
    #     # "/data1/fengjie/datasets/ugi_agent/Task2/citysearch_task4.jsonl",
    #     # "/data1/fengjie/datasets/ugi_agent/Task3/Indicator_Prediction.jsonl",
    #     # "/data1/fengjie/datasets/ugi_agent/Task3/OD_Prediction.jsonl",
    #     # "/data1/fengjie/datasets/ugi_agent/Task3/Site_Selection.jsonl",
    #     "/data1/fengjie/datasets/ugi_agent/Task4/records_final_train_1.17.jsonl",
    #     # "/data1/fengjie/datasets/ugi_agent/WorldModel/wudaokou-gpt4-v2-train-success.jsonl",
    #     # "/data1/fengjie/datasets/ugi_agent/WorldModel/citywalk-wudaokou-mock-train-success.jsonl"
    # ])

    # 面向城市环境内具身智能的模仿virtual-home
    # cwm_dm_data = process_ugi_data(None, data_name="cwm_dm_data", data_path_list=["/data1/fengjie/datasets/ugi_agent/WorldModel/wudaokou-gpt4-v2-train-success.jsonl"])
    # cwm_dm_data_extend = pd.concat([cwm_dm_data]*3, ignore_index=True)

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
        # match = get_best_match("citywalk-wudaokou_small-mock-v10.2-chinese-v3.jsonl", data_path_list)
        # if match is None:
        #     cwm_self_data_navigation = []
        # else:
        #     cwm_self_data_navigation = process_ugi_data(data_path=match, data_name="self-reasoning", data_path_list=None)
        # cwm_self_data_navigation=[]
        cwm_self_data = process_ugi_data(None, data_name="self-reasoning", data_path_list=data_path_list)
        # cwm_self_data = pd.concat([cwm_self_data_navigation]*2 +[cwm_self_data])
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
                "instruction": item[-2]["content"],
                "output": item[-1]["content"],
                "system": "",
                "history":[]
                }
            for i in range(0, int(len(item)/2)*2-2, 2):
            ### 20250224 check一下是否正确，和之前的不太一样：-2
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

# TODO 增加token统计数据

def process_pretrain_data():
    import random
    # data_path = "/data1/fengjie/datasets/pretrain/wikipedia-cn-20230720-filtered/"

    # with open(os.path.join(data_path, "wikipedia-cn-20230720-filtered.json")) as fid:
    #     data = json.load(fid)
    
    # general text
    data_path = "/data1/fengjie/datasets/pretrain/proof-pile-2/sample/"
    data = []
    for file_name in ["arXiv_000.jsonl", "python0000.jsonl", "shard-0000.jsonl"]:
        with jsonlines.open(os.path.join(data_path, file_name)) as fid:
            for i, obj in enumerate(fid):
                if i>=10000: # 控制数量与citygpt大致接近
                    break
                data.append(obj)
    
    # citygpt text
    data_city = []
    with jsonlines.open("/data1/fengjie/datasets/ugi_agent/WorldModel/citywalk-wudaokou-mock-train-success-large-v6.jsonl") as fid:
        for obj in fid:
            # data_city.append(obj)
            pt_text = []
            for i, info in enumerate(obj["diag"]):
                if i==0:
                    pt_text.append(info["content"])
                else:
                    if info["role"] == "assistant":
                        pt_text.append(info["content"])
            data_city.append({"text": "\n".join(pt_text), "source": "citygpt"})


    print("data samples:{}".format(len(data)))
    print("original data:\n{}".format(data[0]))

    data_info = {
        "dataset_name": {
        "file_name": "pt_data_v1.json",
        "ranking": False,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
        }
        }
    }
    print("data_info:{}".format(data_info))

    pt_data = []
    for info in data+data_city:
        item = {
            "instruction": info["text"],
            "input": "",
            "output": "",
            "history": []
        }
        pt_data.append(item)
    print("new data:\n{}".format(pt_data[0]))
    print("data samples:{}".format(len(pt_data)))
    random.shuffle(pt_data)

    # v1 来自proof-pile-2的通用数据和citygpt v12数据的混合版本
    with open(os.path.join("../data/pt_data_v1.json"), "w",  encoding='utf-8') as wid:
        json.dump(pt_data, wid, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # main_ugi()
    # process_pretrain_data()


    # v6-suc: alpaca格式，只使用成功数据，在train数据稳定去除1-shot示例的基础上，增加platypus数据，预期提升榜单效果
    # v6-slim: 降低platypus数据量，同时训练时配合增加cutoff_len为4096来试图提速
    # v7: 增加底层POI数据和路网数据信息的citywalk数据
    # v8: 将decision making数据重复3倍，以期降低后续训练时的epoch需要，增加身份认知数据
    # v9: 将citywalk数据进行扩充，将临近范围由100m扩展为500m，POI限制改为20个
    # v9-200-20-0: 将citywalk数据进行迭代，将临近范围改为200m，POI限制为20个，同时去除了category信息，重复度太高
    # v9-v4: 300m 20个，去除重复路径
    # v9-v5: 增加category，引入grid_xy坐标
    # v10: 去除具身智能任务，专注学习环境信息
    # v11: 将UGI相关数据添加回来，尝试过拟合任务
    # v12: 通用数据+CitiWalk数据（扩展五道口范围，增加相对位置变化描述）
    # v13: 将citywalk数据移动至增量预训练阶段，只保留通用数据
    # v14: 通用数据+扩增后citywalk数据采样部分，保证数据均衡
    # v15: 进一步扩展通用数据，替换sharegpt来源
    # v16: 恢复citywalk空间范围至原本的小范围
    # v17.2-eval: wudaokou_small，配合增加eval上的多类任务以丰富数据多样性
    # v17.3-eval: wudaokou_small，修复eval任务中的数据泄露问题
    # v17.4-eval: 将wudaokou_small评估问题直接应用训练，测试极限场景下的模型记忆能力，理论上应该达到100%的准确率
    # v18.1-address: 增加address数据，丰富数据量
    # v18.2-address-eval: 增加eval数据1遍，尝试过拟合
    # v18.3-eval-repeat: eval数据直接重复5遍，不含answer后缀，回答正确率没有变化
    # v18.4-eval-ans：eval数据增加answer后缀，重复5遍，几乎可以100%回答正确
    # v18.5-eval-ans-1: 降低为1遍，且减少其他数据，回答正确率显著降低
    # v18.6-eval-wangjing: 使用wangjing的eval数据，只增加1遍，其他数据与原先保持一致
    # v18.7-eval-wj5: 使用wangjing的eval数据，增加3遍，其他数据与原先保持一致
    # v19.1-eval: 更新使用v9版本的address和additional数据，citywalk保持v8
    # v19.2-eval: 扩展通用数据量
    # v19.4-eval: 增加基于citywalk构造的推理中间步骤数据，增加外部推理中间步骤数据 
    # v19.5-eval: 进一步扩展基于citywalk自我构造的中间推理步骤，增加知识回忆环节，并将导航任务单独提取出来
    # v19.6-eval: citywalk中增加起始点地址，导航任务形式优化
    # v20.1-eval: 提问模板泛化，补充geoglue数据
    # v21.1-eng-chi: 更新底层citywalk数据，提升语言流畅度和多样性， eng-chi
    # v21.2-chi-chi: chi-chi
    # v21.3-eng-eng: eng-eng
    # v21.4-multi-style-v2: 对基于citywalk自我构造的中间推理步骤进行不同轮数的拓展并进行一定程度泛化
    # v21.5-multi-style: 进一步增加general数据集，确认是否可以继续提高通用表现
    # v21.6-multi-style: 减少general数据集，确认是否模型表现
    # v21.7-multi-style: 增加新的推理数据
    # v21.8-multi-stype: 将多轮推理数据进行增强, 将导航数据扩展3倍
    # v21.4v3-data-ablation-1: 只考虑最最简单的通用数据，platypus+ultralchat+sharepgpt
    # v21.4v3-data-ablation-2: 增加spatial-reasoning，增加agent_instruct
    # v21.4v3-data-ablation-3: 增加cityaddr，含eval2train
    # v21.4v3-data-ablation-4: 增加citywalk，含eval2train
    # v21.4v3-data-ablation-5: 增加citywalk+cityaddress
    # v21.4v3-data-ablation-6: 增加citywalk+cityaddress+self-reasoning, 等价于v21.4-v2
    # v22.4-newyork-multistyle-v6-ny: 替换reasoning数据为ny自己的短途数据
    # v22.4-newyork-multistyle-v6-pr: 替换reasoning数据为paris的短途数据
    # v22.4-paris-multistyle-v6-ny: 替换reasoning数据为ny自己的短途数据
    # v22.4-paris-multistyle-v6-pr: 替换reasoning数据为paris的短途数据
    # v22.4-paris-multistyle-v3-wudaokou: 替换reasoning数据为wudaokou的短途数据
    # v22.4-newyork-multistyle-v3-wudaokou: 替换reasoning数据为wudaokou的短途数据

    # 构造数据时依赖的citygpt专用citywalk数据集名称
    # INPUT_CITYWALK_DATA="citywalk-wudaokou_small-mock-v11-eng-chi.jsonl"
    # INPUT_ADDRESS_DATA="address-wudaokou_small-v11-eng-chi.jsonl"
    # INPUT_SPATIAL_DATA="additional-wudaokou_small-mock-v10.2.jsonl"
    # INPUT_SELF_DATA="direction_reasoning_multistyle_v3.jsonl,distance_reasoning_multistyle_v3.jsonl,citywalk-wudaokou_small-mock-v10.2-chinese-v3.jsonl"
    # INPUT_CITYEVAL_DATA="wangjing/v1/*"
    # # 输出数据的数据集名称，将在后续训练脚本中使用
    # OUTPUT_DATA_NAME_KEY="citygpt-citywalk-v11-cwm-v21.4v3-data-ablation-5"

    # v22.4-newyork-multistyle-v6-ny: 替换reasoning数据为ny自己的短途数据
    # INPUT_CITYWALK_DATA="citywalk-newyork-mock-v11.1-eng-eng.jsonl"
    # INPUT_ADDRESS_DATA="address-newyork-v11.3-eng-eng-selected.jsonl"
    # INPUT_SPATIAL_DATA="additional-wudaokou_small-mock-v10.2.jsonl"
    # INPUT_SELF_DATA="dir_multistyle_newyork_v6.jsonl,dis_multistyle_newyork_v6.jsonl,other_reasoning_newyork_v6.jsonl,citywalk-newyork-mock-v11.1-eng-eng-v3.jsonl"
    # INPUT_CITYEVAL_DATA="wangjing/v1/*"
    # OUTPUT_DATA_NAME_KEY="citygpt-citywalk-v11-cwm-v22.4-newyork-multistyle-v6-ny"

    # # v22.4-newyork-multistyle-v6-pr: 替换reasoning数据为ny自己的短途数据
    # INPUT_CITYWALK_DATA="citywalk-newyork-mock-v11.1-eng-eng.jsonl"
    # INPUT_ADDRESS_DATA="address-newyork-v11.3-eng-eng-selected.jsonl"
    # INPUT_SPATIAL_DATA="additional-wudaokou_small-mock-v10.2.jsonl"
    # INPUT_SELF_DATA="dir_multistyle_paris_v6.jsonl,dis_multistyle_paris_v6.jsonl,other_reasoning_paris_v6.jsonl,citywalk-paris-mock-v11.1-eng-eng-v3.jsonl,citywalk-newyork-mock-v11.1-eng-eng-v3.jsonl"
    # INPUT_CITYEVAL_DATA="wangjing/v1/*"
    # OUTPUT_DATA_NAME_KEY="citygpt-citywalk-v11-cwm-v22.4-newyork-multistyle-v6-pr"

    # v22.4-paris-multistyle-v6-pr: 替换reasoning数据为paris自己的短途数据
    # INPUT_CITYWALK_DATA="citywalk-paris-mock-v11.1-eng-eng.jsonl"
    # INPUT_ADDRESS_DATA="address-paris-v11.3-eng-eng-selected.jsonl"
    # INPUT_SPATIAL_DATA="additional-wudaokou_small-mock-v10.2.jsonl"
    # INPUT_SELF_DATA="dir_multistyle_paris_v6.jsonl,dis_multistyle_paris_v6.jsonl,other_reasoning_paris_v6.jsonl,citywalk-paris-mock-v11.1-eng-eng-v3.jsonl"
    # INPUT_CITYEVAL_DATA="wangjing/v1/*"
    # OUTPUT_DATA_NAME_KEY="citygpt-citywalk-v11-cwm-v22.4-paris-multistyle-v6-pr"

    # v22.4-paris-multistyle-v6-ny: 替换reasoning数据为newyork自己的短途数据
    # INPUT_CITYWALK_DATA="citywalk-paris-mock-v11.1-eng-eng.jsonl"
    # INPUT_ADDRESS_DATA="address-paris-v11.3-eng-eng-selected.jsonl"
    # INPUT_SPATIAL_DATA="additional-wudaokou_small-mock-v10.2.jsonl"
    # INPUT_SELF_DATA="dir_multistyle_newyork_v6.jsonl,dis_multistyle_newyork_v6.jsonl,other_reasoning_newyork_v6.jsonl,citywalk-newyork-mock-v11.1-eng-eng-v3.jsonl,citywalk-paris-mock-v11.1-eng-eng-v3.jsonl"
    # INPUT_CITYEVAL_DATA="wangjing/v1/*"
    # OUTPUT_DATA_NAME_KEY="citygpt-citywalk-v11-cwm-v22.4-paris-multistyle-v6-ny"

    # 构造数据时依赖的citygpt专用citywalk数据集名称
    # INPUT_CITYWALK_DATA="citywalk-newyork-mock-v11.1-eng-eng.jsonl"
    # INPUT_ADDRESS_DATA="address-newyork-v11.3-eng-eng-selected.jsonl"
    # INPUT_SPATIAL_DATA="additional-wudaokou_small-mock-v10.2.jsonl"
    # INPUT_SELF_DATA="direction_reasoning_multistyle_v3.jsonl,distance_reasoning_multistyle_v3.jsonl,citywalk-wudaokou_small-mock-v10.2-chinese-v3.jsonl,citywalk-newyork-mock-v11.1-eng-eng-v3.jsonl"
    # INPUT_CITYEVAL_DATA="wangjing/v1/*"
    # # 输出数据的数据集名称，将在后续训练脚本中使用
    # OUTPUT_DATA_NAME_KEY="citygpt-citywalk-v11-cwm-v22.4-newyork-multistyle-v3-wudaokou"

    INPUT_CITYWALK_DATA="citywalk-NewYork-mock-v15.2.jsonl"
    INPUT_ADDRESS_DATA="address-NewYork-v15.2-selected.jsonl"
    INPUT_SPATIAL_DATA="additional-wudaokou_small-mock-v10.2.jsonl"
    INPUT_SELF_DATA="cityreasoning_SanFrancisco_v15.2-eval.jsonl"
    INPUT_CITYEVAL_DATA="SanFrancisco/v2.3/*"
    # 输出数据的数据集名称，将在后续训练脚本中使用
    OUTPUT_DATA_NAME_KEY="citygpt-NewYork-v24.6-SF-v2.3"


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

    token_path = "/data1/citygpt/init_ckpt/qwen/Qwen1___5-7B-Chat/"
    common_path = "/data1/citygpt/datasets/city_world_model" # 下面有citywalk，merge，general，cache四个路径
    
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
