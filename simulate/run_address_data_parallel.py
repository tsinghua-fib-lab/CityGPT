import os
import numpy as np
import pandas as pd
import argparse
import jsonlines
import subprocess
from tqdm import tqdm

from simulate.address_data import selected_data
from config import DATA_VERSION, REGION_EXP, RESOURCE_PATH

# 注意以下参数暂时不能通过外部赋值，只能在本行修改
PARALLEL_WORKER=20 # 并行模拟数量
###############################################

### 基础参数
MAX_SAMPLES_PER_CATEGORY = 2200 # 每类对话最多抽取的样本数 

### 定义 输入、输出文件名
INPUT_FILE="simulate/tasks/input_citywalk_{}-{}.jsonl".format(REGION_EXP, DATA_VERSION)
OUTPUT_FILE="simulate/examples/address-{}-{}.jsonl".format(REGION_EXP, DATA_VERSION)
POI_FILE = os.path.join(RESOURCE_PATH, "{}_pois.csv".format(REGION_EXP))
AOI_FILE = os.path.join(RESOURCE_PATH, "{}_aois.csv".format(REGION_EXP))
ROAD_FILE = os.path.join(RESOURCE_PATH, "{}_roads.csv".format(REGION_EXP))
SELECTED_OUTPUT_FILE = "simulate/examples/address-{}-{}-selected.jsonl".format(REGION_EXP, DATA_VERSION)

# 切割train_task文件
input_files = [INPUT_FILE.replace("tasks/", "logs/")+"-split-{}.jsonl".format(i) for i in range(PARALLEL_WORKER)]
output_files = [OUTPUT_FILE+"-split-{}.jsonl".format(i) for i in range(PARALLEL_WORKER)]
def task_partition_jsonl(input_file=INPUT_FILE, input_files=input_files):
    tasks = []
    with jsonlines.open(input_file) as fid:
        for i, task_info in enumerate(fid):
            tasks.append(task_info)
    chunk_size, remainder = len(tasks) // PARALLEL_WORKER, len(tasks) % PARALLEL_WORKER
    tasks_parallel = [tasks[i*chunk_size:(i+1)*chunk_size] for i in range(PARALLEL_WORKER)]
    if remainder>0:
        tasks_parallel[-1] += tasks[-remainder:]
    
    for i, input_file in enumerate(input_files):
        with jsonlines.open(input_file, "w") as wid:
            wid.write_all(tasks_parallel[i])

task_partition_jsonl(INPUT_FILE, input_files)

# 切割POI/AOI文件
input_poi_files = [POI_FILE.replace(RESOURCE_PATH, "simulate/logs/")+"-split-{}.jsonl".format(i) for i in range(PARALLEL_WORKER)]
input_aoi_files = [AOI_FILE.replace(RESOURCE_PATH, "simulate/logs/")+"-split-{}.jsonl".format(i) for i in range(PARALLEL_WORKER)]
input_road_files = [ROAD_FILE.replace(RESOURCE_PATH, "simulate/logs/")+"-split-{}.jsonl".format(i) for i in range(PARALLEL_WORKER)]
def task_partition_csv(input_file=AOI_FILE, input_files=input_aoi_files):
    tasks = pd.read_csv(input_file)
    split_tasks = np.array_split(tasks, PARALLEL_WORKER)
    for i, task in enumerate(split_tasks):
        task.to_csv(input_files[i])

task_partition_csv(POI_FILE, input_poi_files)
task_partition_csv(AOI_FILE, input_aoi_files)
task_partition_csv(ROAD_FILE, input_road_files)

# 准备参数
commands = []
for i in range(PARALLEL_WORKER):
    command = ["python", "-m", "simulate.address_data"]
    command.append("--input_file")
    command.append(input_files[i])
    command.append("--output_file")
    command.append(output_files[i])
    command.append("--input_pois_file")
    command.append(input_poi_files[i])
    command.append("--input_aois_file")
    command.append(input_aoi_files[i])
    command.append("--input_roads_file")
    command.append(input_road_files[i])
    commands.append(command)

# 并行执行
processes = []
for i, c in enumerate(commands):
    if len(commands)<=5:
        print("start task:{} command:{}".format(i, c))
    processes.append(subprocess.Popen(c))

for process in tqdm(processes, desc="执行并行任务"):
    process.wait()
print("所有进程执行完成,开始整合文件")

# 合并文件
with jsonlines.open(OUTPUT_FILE, "w") as wid:
    for output_file in output_files:
        with jsonlines.open(output_file, "r") as fid:
            for item in fid:
                wid.write(item)

# 删除临时文件
for file in input_files+output_files+input_poi_files+input_aoi_files+input_road_files:
    os.remove(file)

# 去除GeoQA问题，随机采样数据以降低数据量
selected_data(OUTPUT_FILE, SELECTED_OUTPUT_FILE, MAX_SAMPLES_PER_CATEGORY)
