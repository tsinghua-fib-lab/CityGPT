import copy
import os
import random
import re
import time
import pandas as pd
import numpy as np
import subprocess
import httpx
from openai import OpenAI
from math import sin, cos, atan2, pi

from pycitydata.map import Map
from pycitysim.routing import RoutingClient

from config import MONGODB_URI, PROXY,LLM_MODEL_MAPPING, INFER_SERVER, SERVER_IP, LOCAL_MODEL_KEY

OPENAI_APIKEY = os.environ["OpenAI_API_KEY"]
DEEPINFRA_APIKEY = os.environ["DeepInfra_API_KEY"]
SILICONFLOW_APIKEY = os.environ["SiliconFlow_API_KEY"]
DEEPSEEK_APIKEY = ""
DEEPBRICKS_APIKEY = os.environ["DeepBricks_API_KEY"]

primary_directions = ['east', 'south', 'west', 'north']
secondary_directions = ['southeast', 'northeast', 'southwest', 'northwest']
EW = {'east', 'west'}
NS = {'south', 'north'}
dir_map = {"north": "south-north", "south": "south-north", "west": "east-west", "east": "east-west"}
dir_map2 = {"south-north": "east-west", "east-west": "south-north"}

secondary_dir_to_primary_dirs = {
    "southeast": ("south", "east"),
    "northeast": ("north", "east"),
    "northwest": ("north", "west"),
    "southwest": ("south", "west"),
}


def task_files_adaption(task_file, output_path):
    task_files = copy.deepcopy(task_file)
    path_prefix = output_path
    for k in task_files:
        for kk in task_files[k]:
            if path_prefix not in task_files[k][kk]:
                task_files[k][kk] = os.path.join(path_prefix, task_files[k][kk])
    os.makedirs(path_prefix, exist_ok=True)
    return task_files


def gen_options(options, question, answer):
    """
    生成字典:
    {"A": "选项1", "B": "选项2”, ..., "question": $question$, "answer": answer对应的选项}
    """
    options = copy.copy(options)
    random.shuffle(options)
    result = {}
    start_option_ascii = ord('A')
    for index, option in enumerate(options):
        selection = chr(start_option_ascii + index)
        result[selection] = option
        if option == answer:
            result["answer"] = selection

    if "answer" not in result:
        raise LookupError("未找到option=answer")

    result["question"] = question

    return result

def save_data(unseen_aois, save_path):
    unseen_aois_df = pd.DataFrame(data=unseen_aois)
    unseen_aois_df["is_seen"] = False

    task_df = pd.concat([unseen_aois_df])
    task_df.to_csv(save_path)

def dir_all_dis(routes, secondary_directions, primary_directions,secondary_dir_to_primary_dirs):
    """
    计算输入数据包含的移动方向以及每个方向移动的总距离
    """
    distances = []
    directions = []
    dir_dis_dep = []
    dir_dis = []
    for cnt, route in enumerate(routes):
        if route['type'] == "junc":
            continue
        distance = route['road_length']
        direction = route['direction'].split()[-1]
        distances.append(distance)
        directions.append(direction)

    for cnt2, direction in enumerate(directions):
        if direction in secondary_directions:
            distance = int(distances[cnt2]) * 0.7
            distance_str = str(distances[cnt2]) + "m,equals to ({},{}m) and ({},{}m)".format(direction[0],
                                                                                        "0.7*" + str(distances[
                                                                                            cnt2]) + '=' + str(
                                                                                            distance), direction[1],
                                                                                        "0.7*" + str(distances[
                                                                                            cnt2]) + '=' + str(
                                                                                            distance))
            dir_dis_dep.append((secondary_dir_to_primary_dirs[direction][0], distance))
            dir_dis_dep.append((secondary_dir_to_primary_dirs[direction][1], distance))
            dir_dis.append((direction, distance_str))
        elif direction in primary_directions:
            distance = int(distances[cnt2])
            distance_str = str(distances[cnt2]) + 'm'
            dir_dis_dep.append((direction, distance))
            dir_dis.append((direction, distance_str))
        else:
            print(direction)

    # 遍历原始列表，将相同键值的元素进行累加处理
    mid = {}
    for cnt, ddlist in enumerate(dir_dis_dep):
        dir, dis = ddlist
        if dir not in mid:
            mid[dir] = 0
        mid[dir] += dis
    dir_dis_fin = [(key, value) for key, value in mid.items()]  # 起始点到终点各个方向位移距离，[(方向，位移距离),(),...]
    dirs = set()
    for dir, dis in dir_dis_fin:
        dirs.add(dir)
    short_dir = list(set(primary_directions).difference(dirs))
    if len(short_dir) > 0:
        for dir in short_dir:
            dir_dis_fin.append((dir, 0))
    return dir_dis_fin, dir_dis

def compute_length(routine):  # 计算导航路径的总长度
    float_values = re.findall(r'for (\d+) meters', routine)
    length = np.sum([int(num) for num in float_values])
    return length

# 计算点1到点2的方向角
def calcu_azimuth(lat1, lon1, lat2, lon2):
    lat1_rad = lat1 * pi / 180
    lon1_rad = lon1 * pi / 180
    lat2_rad = lat2 * pi / 180
    lon2_rad = lon2 * pi / 180
    y = sin(lon2_rad - lon1_rad) * cos(lat2_rad)
    x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(lon2_rad - lon1_rad)
    brng = atan2(y, x) / pi * 180
    return float((brng + 360.0) % 360.0)


def angle2dir(angle):
    Direction = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
    s = 22.5
    for i in range(8):
        if angle < s + 45 * i:
            return Direction[i]
    return Direction[0]


def angle2dir_4(angle):
    Direction = ['north', 'east', 'south', 'west']
    if angle < 45 or angle >= 315:
        return Direction[0]
    elif 45 <= angle < 135:
        return Direction[1]
    elif 135 <= angle < 225:
        return Direction[2]
    else:
        return Direction[3]



def extract_choice(gen, choice_list):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^A-H]{0,20}?(?:n't|not))[^A-H]{0,10}?\b(?:|is|:|be))\b)[^A-H]{0,20}?\b([A-H])\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b([A-H])\b(?![^A-H]{0,8}?(?:n't|not)[^A-H]{0,5}?(?:correct|right))[^A-H]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^([A-H])(?:\.|,|:|$)", gen)

    if res is None:
        res = re.search(r"\n\s*([A-H])", gen)

    if res is None:
        return "Z"
    
    return res.group(1)


def get_chat_completion(session, model_name, max_tokens=1200, temperature=0, infer_server=None):
    client = get_llm_model_client(model_name, infer_server)
    # 统一--传进来的是model_name
    model_name = LLM_MODEL_MAPPING[model_name]
    MAX_RETRIES = 3
    WAIT_TIME = 1
    for i in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=session,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if i < MAX_RETRIES - 1:
                time.sleep(WAIT_TIME)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "OpenAI API Error."
            

def get_llm_model_client(model_name, infer_server=None):
    if infer_server is None:
        for server_name in INFER_SERVER:
            if model_name in INFER_SERVER[server_name]:
                infer_server=server_name
                break
    # 统一--传进来的是model_name
    try:
        model_name = LLM_MODEL_MAPPING[model_name]
    except:
        print(f"Our train model")

    if infer_server=='OpenAI':
        client = OpenAI(
            http_client=httpx.Client(proxy=PROXY),
            api_key=OPENAI_APIKEY
            )
    elif infer_server =="DeepInfra":
        client = OpenAI(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key=DEEPINFRA_APIKEY,
        http_client=httpx.Client(proxies=PROXY),
            )
    elif infer_server =="Siliconflow":
        client = OpenAI(
        api_key=SILICONFLOW_APIKEY,
        base_url="https://api.siliconflow.cn/v1"
        )
    else:
        model_name, port = model_name.split(":")
        client = OpenAI(
            base_url="http://{}:{}/v1".format(SERVER_IP, port),
            api_key=LOCAL_MODEL_KEY
        )

    return client

def load_map(city_map, cache_dir, routing_path, port):
    m = Map(
            mongo_uri=f"{MONGODB_URI}",
            mongo_db="llmsim",
            mongo_coll=city_map,
            cache_dir=cache_dir,
        )
    route_command = f"{routing_path} -mongo_uri {MONGODB_URI} -map llmsim.{city_map} -cache {cache_dir} -listen localhost:{port}"
    cmd = route_command.split(" ")
    print("loading routing service")
    process = subprocess.Popen(args=cmd, cwd="./")
    routing_client = RoutingClient(f"localhost:{port}")
    return m, process, routing_client
