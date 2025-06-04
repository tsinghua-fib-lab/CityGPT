import argparse
import jsonlines
import copy
import pandas as pd

from simulate.prompts import get_system_instruction, get_prompts
from simulate.agent import inject_info
from simulate.utils import RunMode
from simulate.templates import navigation_choose
from config import EVAL_DATA, LANDMARK_DATA

def main(log_file_name, output_file_name):
    """从历史日志中拼接出对话数据，原版数据忘记存储……"""
    data = []
    with jsonlines.open("simulate/logs/{}".format(log_file_name)) as fid:
        for obj in fid:
            info = [obj["done"], obj["final_reason"], obj["task"]]
            data.append(info)
    data = pd.DataFrame(data=data, columns=["done", "reason", "task"])
    data[data["done"]==True]

    data = []
    system_prompt = get_system_instruction()
    example = get_prompts("buy")
    example[0] = "Here is one example.\n" + example[0]

    with jsonlines.open("simulate/logs/{}".format(log_file_name)) as fid:
        for i, obj in enumerate(fid):
            if obj["done"]:
                session = []

                session.append({"role": "user", "content": system_prompt})
                session.append({"role": "assistant", "content": "OK. I'll follow your instructions and try my best to solve the task."})

                session = inject_info(session, example)

                init_prompt = obj["log_info"]["init_prompt"]
                session.append({"role": "user", "content": init_prompt})

                history = obj["log_info"]["log"]
                for item in history:
                    llm_output = item["output"]
                    env_output = item["observation"]
                    session.append({"role": "assistant", "content": llm_output})
                    session.append({"role": "user", "content": env_output})

                data.append(
                    {
                        "task":"buy",
                        "id": log_file_name+"-"+str(i),
                        "diag": session
                    }
                )

    with jsonlines.open("./examples/{}".format(output_file_name), "w") as wid:
        for d in data:
            wid.write(d)


def simple_main(log_file_name, output_file_name, filter=False, task_type=RunMode.NORMAL.value, one_round=False):
    """从历史记录直接抽取session数据"""
    data = []

    # 读取数据
    with jsonlines.open(log_file_name) as fid:
        for i, obj in enumerate(fid):
            if filter:
                flag = obj["done"]
            else:
                flag = True

            if flag:
                if "session" not in obj["log_info"]:
                    continue
                
                # 去除1-shot样例，只保留init_prompt和其他交互数据
                compress_session = []
                data_should_be_left = True
                for ss in obj["log_info"]["session"]:

                    # 遇到1-shot标记开始，遇到新任务表示结束
                    if "Here is one example." in ss["content"]:
                        data_should_be_left = False
                    if "Here is your task" in ss["content"]:
                        data_should_be_left = True
                    if ss["role"] == "image_paths":
                        data_should_be_left = True
                    
                    if data_should_be_left:
                        compress_session.append(ss)
                
                if task_type==RunMode.CITY_WALK.value:
                    compress_session2 = citywalk_clean(copy.deepcopy(compress_session), one_round)
                
                data.append(
                    {
                        "task": obj["type"],
                        "id": log_file_name+"-"+str(i),
                        "diag": compress_session2
                    }
                )
    
    # 路径去重
    route_dict = {}
    for i, item in enumerate(data):
        route = item["diag"][1]["content"]
        if route not in route_dict:
            route_dict[route] = [i]
        else:
            route_dict[route].append(i)
    
    data_deduplicated = []
    for route in route_dict:
        item_id = route_dict[route][0]
        data_deduplicated.append(data[item_id])

    # 保存数据
    with jsonlines.open(output_file_name, "w") as wid:
        for d in data_deduplicated:
            wid.write(d)


def citywalk_clean(session, one_round=False):
    """将citywalk得到的训练数据进行格式调整和清理"""

    init_prompt = session.pop(0)
    assert "You are a tourist" in init_prompt["content"], "去除init_prompt，训练citygpt时无用"
    session.pop(0)


    # 去除AVAILABLE ACTIONS
    for sess in session:
        if sess["role"] == "user":
            sess["content"] = sess["content"].split("AVAILABLE ACTIONS")[0]
    
    # 去除navigate字段，将user和assistant互转
    traverse = {"user": "assistant", "assistant": "user"}
    for i, sess in enumerate(session):
        if sess["role"] == "image_paths":
            # print(sess["content"])
            continue
        if i==0:
            assert sess["role"] == "user" and ("Here is your task" in sess["content"] or "这是您的任务" in sess["content"]), "保留task描述作为session开始"
            continue
        elif i==1:
            continue
        else:
            # 将回答中的第一句中的实际移动行为描述转移至问题中
            if sess["role"]=="assistant":
                sess["content"] = session[i+1]["content"].split("\n")[0]
            else:
                if EVAL_DATA == True:
                    pass
                else:
                    sess["content"] = "After "+ sess["content"]
            sess["role"] = traverse[sess["role"]]
    
    # 去除无用的assistant问题
    assert session[1]["role"] == "assistant" and "navigate" in session[1]["content"], "去除第一个assistant问题"
    # print(session[1]["content"])
    del session[1]

    
    if one_round:
        first_user = None
        first_assistant = None
        image_paths = None
        for sess in session:
            if sess["role"] == "image_paths":
                # print(sess["content"])
                image_paths = sess
            if sess["role"] == "user" and first_user is None:
                if EVAL_DATA == False:
                    original_content = sess["content"]
                    # 删除 "Here is your task." 和位置信息
                    if "Here is your task." in original_content:
                        task_part = original_content.split("Here is your task. ")[1]
                        position_info_start = task_part.find("Your current position is")
                        task_description = task_part[:position_info_start].strip()

                        # 提取起终点poi
                        parts = task_description.split(" and you need to go to ")
                        if len(parts) == 2:
                            start_location = parts[0].replace("You are in ", "")
                            destination = parts[1].strip(". ") 
                            # 重新生成user的content
                            new_content = navigation_choose(start_location, destination)
                            sess["content"] = new_content  

                first_user = sess  
            elif sess["role"] == "assistant" and first_user is not None and first_assistant is None:
                first_assistant = sess
        

        if first_user and first_assistant:
            if image_paths:
                return [first_user, first_assistant, image_paths]
            else:
                return [first_user, first_assistant]
    else:
        return session
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="simulate/logs/output_citywalk_wudaokou_small_mock_100_10_1_v11.1-eng-chi-v3.jsonl")
    parser.add_argument("--sft_file", type=str, default="simulate/examples/citywalk-wudaokou_small-v11.1-eng-chi-v3.jsonl")
    
    args = parser.parse_args()

    # 决定数据是否只包含一轮对话
    if LANDMARK_DATA:
        one_round = True
    else:
        one_round = False

    simple_main(
        log_file_name=args.log_file,
        output_file_name=args.sft_file,
        filter=True,
        task_type=RunMode.CITY_WALK.value,
        one_round=one_round
        )
