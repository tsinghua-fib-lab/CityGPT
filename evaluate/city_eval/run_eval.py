import os
import sys
import tqdm
import jsonlines
import argparse
import pandas as pd
import time
import re
from thefuzz import process
from multiprocessing import Pool
import copy
from openai import OpenAI
import httpx
import random

from config import EVAL_TASK_MAPPING_v1, EVAL_TASK_MAPPING_v2, REGION_EXP
from .utils import get_chat_completion, extract_choice


def task_files_adaption(task_file, path_prefix):
    for category, tasks in task_file.items():
        for task_name, file_name in tasks.items():
            task_file[category][task_name] = os.path.join(path_prefix, file_name)
    os.makedirs(path_prefix, exist_ok=True)
    return task_file

FEW_SHOT_PROMPT_SHORT_DIR = """
Here is an example:
Question: After Starting from 第三食堂小卖部,you walk along 王庄路 for 800 meters from south to north, then go east towards the intersection of 王庄路 and 清华东路, walk along 清华东路 for 1000 meters from west to east, then go east towards the intersection of 清华东路 and 清华东路, walk along 清华东路 for 700 meters from west to east, then go east towards the intersection of 清华东路 and 清华东路, walk along 清华东路 for 200 meters from west to east, then go southeast towards the intersection of 清华东路 and 志新西路, walk along 志新西路 for 800 meters from northwest to southeast.Finally you arrive at 诚和敬二里庄军休所养老驿站.The address of 第三食堂小卖部 is 王庄路东侧, 距离成府路和王庄路交叉口北角250m,the address of 诚和敬二里庄军休所养老驿站 is 志新西路东侧, 距离小月河西路和志新西路交叉口南角350m.In which direction is 诚和敬二里庄军休所养老驿站 from 第三食堂小卖部?\nA. 北\nB. 东\nC. 西\nD. 南\n Let's think step by step.
Answer:B
Step 1: Determine the direction and distance of each segment of the journey in the form of "The direction you are facing when walking along the road(distance)":\n north (800m),east (1000m),east (700m), east (200m),southeast (800m),equals to south(800*0.7=560m),east(800*0.7=560m)
Step 2: Analyze the overall direction of the journey.\nCalculate the total length of each direction:\neast:1000+700+200+560=2460.west:0.north:800.south:560.\n\nStep 3:800 is larger than 560,so there is a northbound travel.2460 is larger than 0,so there is a eastbound travel.\n\nStep 4:Compare above two directions,south:800-560=240，east:2460-0=2460.2460 is larger than 240,so the overall direction is east.So the answer is east.\nAnswer:B
"""

FEW_SHOT_PROMPT_SHORT_DIS = """
Here is an example:
Q:After Starting from 城华园招待所(学清路西侧学院路周边内, 距离月泉路和学清路交叉口南角400m), you walk along 学清路 for 700 meters 从北到南, then go 从北到南 towards 学清路和学清路交叉口, walk along 学清路 for 600 meters 从北到南, then go 从北到南 towards 学清路和学院路交叉口, walk along 学院路 for 100 meters 从北到南, then go 从北到南 towards 学院路和学院路交叉口, walk along 学院路 for 500 meters 从北到南, then go 从东到西 towards 学院路和成府路交叉口, walk along 成府路 for 1000 meters 从东到西, then go 从南到北 towards 成府路和王庄路交叉口, walk along 王庄路 for 800 meters 从南到北 .Finally you arrive at 北京语言大学家属区停车场-出入口(王庄路东侧, 距离成府路和王庄路交叉口东北角200m).How many meters do I need to walk from 城华园招待所(学清路西侧学院路周边内, 距离月泉路和学清路交叉口南角400m) to 北京语言大学家属区停车场-出入口(王庄路东侧, 距离成府路和王庄路交叉口东北角200m) along the road?\nA. 3700.0\nB. 2700.0\nC. 1850.0\nD. 7400.0\n
A:Please think step by step.
Step 1: Find the distance traveled for each segment of the road：700m,600m,100m,500m,1000m，800m
Step 2:Add all distances above together to get the total distance:her number:700+600+100+500+1000+800=3700.So the answer is 3700.\nAnswer:A.
"""
FEW_SHOT_PROMPT_LONG_DIR = """
Here is an example:
Q:After Starting from 第三食堂小卖部,you walk along 王庄路 for 800 meters from south to north, then go east towards the intersection of 王庄路 and 清华东路, walk along 清华东路 for 1000 meters from west to east, then go east towards the intersection of 清华东路 and 清华东路, walk along 清华东路 for 700 meters from west to east, then go east towards the intersection of 清华东路 and 清华东路, walk along 清华东路 for 200 meters from west to east, then go southeast towards the intersection of 清华东路 and 志新西路, walk along 志新西路 for 800 meters from northwest to southeast.Finally you arrive at 诚和敬二里庄军休所养老驿站.The address of 第三食堂小卖部 is 王庄路东侧, 距离成府路和王庄路交叉口北角250m,the address of 诚和敬二里庄军休所养老驿站 is 志新西路东侧, 距离小月河西路和志新西路交叉口南角350m.In which direction is 诚和敬二里庄军休所养老驿站 from 第三食堂小卖部?\nA. 北\nB. 东\nC. 西\nD. 南\n
A: Let's think step by step.
Step 1: Determine the direction of each segment of the journey.\nFrom 第三食堂小卖部 to the intersection of 王庄路 and 清华东路: north (800m)\nFrom the intersection of 王庄路 and 清华东路 to the intersection of 清华东路 and 清华东路: east (1000m)\nFrom the intersection of 清华东路 and 清华东路 to the intersection of 清华东路 and 清华东路: east (700m)\nFrom the intersection of 清华东路 and 清华东路 to the intersection of 清华东路 and 志新西路: east (200m)\nFrom the intersection of 清华东路 and 志新西路 to 诚和敬二里庄军休所养老驿站: southeast (800m),equals to south(800*0.7=560m),east(800*0.7=560m)\n\n
Step 2: Analyze the overall direction of the journey.\nCalculate the total length of each direction:\nFirst find out the distances of each direction:east:1000,700,200,560,west:0,north:800,south:560.\nThen add two of the numbers of eastern direction:1000+700=1700.\nThen add the sum with another number of eastern direction:1700+200=1900.\nThen add the sum with last number of eastern direction:1900+560=2460.So the total length of each direction:east:2460,west:0,north:800,south:560\n\n
Step 3:800 is larger than 560,so there is a northbound travel.2460 is larger than 0,so there is a eastbound travel.\n\n
Step 4:Compare above two directions,south:800-560=240，east:2460-0=2460.2460 is larger than 240,so the overall direction is east.So the answer is east.\nAnswer:B
"""


INIT_PROMPT = "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question. Please output the option directly. No need for explaination.\n"


def format_example(line, include_answer=True, max_choices=4, fewshot=FEW_SHOT_PROMPT_LONG_DIR, inlcude_answer_prompt_final=True, multi_round=False, task_name=""):
    if multi_round:
        # 多轮复杂推理评测时弃用简单的多选设置
        prompt = fewshot
    else:
        # 单轮简单推理可增加该prompt输出简单答案
        prompt = INIT_PROMPT + fewshot
    # choices = ["A", "B", "C"]
    choices = ["A", "B"]
    if max_choices>=3:
        choices.append("C")
    if max_choices>=4:
        choices.append("D")
    if max_choices>=5:
        choices.append("E")
    if max_choices>=6:
        choices+=["F","G","H"]
    example = prompt + 'Question: ' + line['question']
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'
    
    if include_answer:
        example += '\nAnswer: ' + line["answer"] + '\n\n'
    else:
        if inlcude_answer_prompt_final:
            # 增加该参数，模型会倾向于直接输出简单答案
            example += '\nAnswer:'
        pass
    return example 

###################### 评估接口
def run_evaluate_api(task_file_path, model_name, max_validation, temperature, max_tokens, fewshot, inlcude_answer_prompt_final, multi_round, region_exp="wudaokou_small", evaluate_version="v1"):
    test_df = pd.read_csv(task_file_path, header=0)

    columns = test_df.columns.to_list()
    if "H" in columns:
        max_choices = 8
    elif "E" in columns:
        max_choices = 5
    elif "D" in columns:
        max_choices = 4
    elif "C" in columns:
        max_choices = 3
    else:
        max_choices = 2
    
    # 只随机取一部分进行测试，设置随机种子保证可复现
    if test_df.shape[0]>max_validation*2:
        test_df = test_df.sample(max_validation, random_state=42)
    # test_df = test_df.sample(max_validation, random_state=42)
    
    correct_count, count = 0, 0
    res = []
    for _, row in tqdm.tqdm(test_df.iterrows(), total=len(test_df)):
        # 单轮简单评测
        if not multi_round:
            question = format_example(
                row, 
                include_answer=False, 
                max_choices=max_choices, 
                fewshot=fewshot, 
                inlcude_answer_prompt_final=inlcude_answer_prompt_final,
                multi_round=multi_round,
                task_name=task_name
                )
            question = question[:2000]
            output = get_chat_completion(
                session=[{"role":"user", "content": question}], 
                model_name=model_name, temperature=temperature, max_tokens=max_tokens
                )
            # 20241017增加
            if len(output) == 0:
                extract = "No output"
            else:
                extract = extract_choice(output, ["A", "B", "C", "D", "E", "F","G","H"]) 

            res.append([{"role":"user", "content": question}, {"role":"assistant", "content": output}, {"role":"ref", "content": row["answer"]}, {"role":"extract", "content": extract}])
        # 多轮复杂推理评测
        else:
            question1 = format_example(
                row, 
                include_answer=False, 
                max_choices=max_choices, 
                fewshot="", 
                inlcude_answer_prompt_final=False,
                multi_round=multi_round,
                task_name=task_name
                )
            question1 = question1[:1900]
            round1 = [{"role":"user", "content": question1+"\nTo answer the question, please think about the navigation instruction about two locations."}]
            output1 = get_chat_completion(
                session=round1, 
                model_name=model_name, temperature=temperature, max_tokens=max_tokens
                )

            round2 = round1 + [
                {"role":"assistant", "content": "The navigation instruction: "+output1}, 
                {"role":"user", "content": "Please consider the above context and answer the following question. " + question1 + "\nLet's think step by step."}
                ]
            output2 = get_chat_completion(
                session=round2, 
                model_name=model_name, temperature=temperature, max_tokens=max_tokens
                )
            round3 = round2 + [
                {"role":"assistant", "content": output2},
                {"role":"user", "content": "Thus, " + question1 + "\nAnswer:" if inlcude_answer_prompt_final else ""}
            ]
            output = get_chat_completion(
                session=round3, 
                model_name=model_name, temperature=temperature, max_tokens=max_tokens
                )
            
            # 20241017增加
            if len(output) == 0:
                extract = "No output"
            else:
                extract = extract_choice(output, ["A", "B", "C", "D", "E", "F","G","H"]) 

            round_final = round3 + [{"role":"aisstant", "content": output}, {"role":"ref", "content": row["answer"]}, {"role":"extract", "content": extract}]
            
            res.append(round_final)

        if len(output) == 0:
            pass
        else:
            ans = extract_choice(output, ["A", "B", "C", "D", "E", "F","G","H"]) 
            if ans==row["answer"]:
                correct_count += 1
        count += 1
    
    print("Success rate:{}({}/{})".format(correct_count/count, correct_count, count))

    os.makedirs("evaluate/city_eval/results/logs_{}/".format(evaluate_version), exist_ok=True)
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    with jsonlines.open("evaluate/city_eval/results/logs_{}/city_eval_{}_{}_{}_{}_{}.jsonl".format(evaluate_version, region_exp, model_name, task_name, "multiround" if multi_round else "singleround", inlcude_answer_prompt_final), "w") as wid:
        for r in res:
            wid.write(r)
    return [model_name, task_name, correct_count, count, correct_count/count]


if __name__ == "__main__":
    print("开始进行模型评估_run-eval-v2")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="chatglm3-v21.4:23130")
    parser.add_argument("--city_eval_version", type=str, default="v1")
    parser.add_argument("--max_tokens", default=500, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_valid", type=int, default=1)
    parser.add_argument("--fewshot", action="store_true")
    parser.add_argument("--include_answer_prompt_final", action="store_true", help="在提问的最后是否添加Answer后缀，默认不加，如需增加在参数中增加该命令即可")
    parser.add_argument("--multi_round", action="store_true", help="启用面向多轮复杂推理的参数，不设置即不启用，设置即启用")
    parser.add_argument("--auto_multi", action="store_true", help="基于规则自动选择是否多轮，启用时会覆盖multi_round参数")
    parser.add_argument("--workers", type=int, default=1, help="与eval_parallel共同使用, 决定进程数")
    args = parser.parse_args()
    res_df = []

    # 定义测试方式，是否使用fewshot
    fewshot_mapping = {
        "dis": FEW_SHOT_PROMPT_SHORT_DIS,
        "dir": FEW_SHOT_PROMPT_SHORT_DIR
        }
    print("testing mode:", "fewshot" if args.fewshot else "zeroshot")

    if REGION_EXP == "wudaokou_small" or REGION_EXP == "wudaokou_large" or REGION_EXP == "yuyuantan" or REGION_EXP == "dahongmen" or REGION_EXP == "beijing" or REGION_EXP == "wangjing":
        EVAL_TASK_MAPPING = EVAL_TASK_MAPPING_v2
    else:
        if args.city_eval_version == "v1":
            EVAL_TASK_MAPPING = EVAL_TASK_MAPPING_v1
        elif args.city_eval_version == "v2":
            EVAL_TASK_MAPPING = EVAL_TASK_MAPPING_v2

    task_path = "evaluate/city_eval/tasks/{}/{}".format(REGION_EXP, args.city_eval_version)
    NEW_TASK_FILES = task_files_adaption(EVAL_TASK_MAPPING, task_path)

    if args.workers>1:
        # 多进程测试
        para_group = []
        for model in [args.model_name]: 
            for category, tasks in NEW_TASK_FILES.items():
                for task_name, file_path in tasks.items():
                    print("evaluate model:{} task:{}".format(model, file_path))

                    # fewshot示例
                    if not args.fewshot:
                        few_shot_str = ""
                    else:
                        if "dis" in file_path:
                            few_shot_str = fewshot_mapping["dis"]
                        elif "dir" in file_path:
                            few_shot_str = fewshot_mapping["dir"]
                        else:
                            raise NotImplementedError
                    
                    # 有上下文时直接单轮作答，没有上下文时多轮推理
                    if args.auto_multi:
                        if "noroutine" in file_path:
                            args.multi_round = True
                        elif "routine" in file_path:
                            args.multi_round = False
                        else:
                            args.multi_round = False
                    
                    para_group.append((
                        file_path,
                        model,
                        args.max_valid,
                        args.temperature,
                        args.max_tokens,
                        few_shot_str,
                        args.include_answer_prompt_final,
                        args.multi_round,
                        REGION_EXP,
                        args.city_eval_version))

        with Pool(args.workers) as pool:
            results = pool.starmap(run_evaluate_api, para_group)
        for res in results:
            res_df.append(res)
    else:
        # 单进程测试
        for model in [args.model_name]: 
            for category, tasks in NEW_TASK_FILES.items():
                for task_name, file_path in tasks.items():
                    print("evaluate model:{} task:{}".format(model, file_path))

                    # fewshot示例
                    if not args.fewshot:
                        few_shot_str = ""
                    else:
                        if "dis" in file_path:
                            few_shot_str = fewshot_mapping["dis"]
                        elif "dir" in file_path:
                            few_shot_str = fewshot_mapping["dir"]
                        else:
                            raise NotImplementedError
                    # 有上下文时直接单轮作答，没有上下文时多轮推理
                    if args.auto_multi:
                        if "noroutine" in file_path:
                            args.multi_round = True
                        elif "routine" in file_path:
                            args.multi_round = False
                    
                    res = run_evaluate_api(
                        file_path,
                        model,
                        args.max_valid,
                        args.temperature,
                        args.max_tokens,
                        fewshot=few_shot_str,
                        inlcude_answer_prompt_final=args.include_answer_prompt_final,
                        multi_round=args.multi_round,
                        region_exp=REGION_EXP,
                        evaluate_version=args.city_eval_version
                    )
                    res_df.append(res)

    if "/" in args.model_name:
        model_name = args.model_name.split("/")[-1]
    else:
        model_name = args.model_name.replace(":", "-")
    res_df = pd.DataFrame(res_df, columns=["model_name", "task_name", "corrct", "count","accuracy"])
    res_df.to_csv("evaluate/city_eval/results/city_eval_{}_{}_{}_{}_{}_{}.csv".format(
        REGION_EXP,
        args.city_eval_version, 
        model_name, 
        "fewshot" if args.fewshot else "zeroshot", 
        "multiround" if args.multi_round else "singleround",
        args.include_answer_prompt_final
        ))
