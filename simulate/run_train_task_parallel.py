import os
import numpy as np
import pandas as pd
import argparse
import jsonlines
import subprocess
from tqdm import tqdm

from config import DATA_VERSION, REGION_EXP, RESOURCE_PATH

# 注意以下参数暂时不能通过外部赋值，只能在本行修改
PARALLEL_WORKER=30 # 并行模拟数量
###############################################

### 基础参数
### 定义 输入、输出文件名
OUTPUT_FILE="simulate/tasks/input_citywalk_{}-{}.jsonl".format(REGION_EXP, DATA_VERSION)
AOI_FILE = os.path.join(RESOURCE_PATH, "{}_aois.csv".format(REGION_EXP))
ROAD_INFO_FILE = "simulate/logs/roads_info_{}-{}.csv".format(REGION_EXP, DATA_VERSION)

# 切割aoi文件
output_files = [OUTPUT_FILE+"-split-{}.jsonl".format(i) for i in range(PARALLEL_WORKER)]

# 切割POI/AOI文件
input_aoi_files = [AOI_FILE.replace(RESOURCE_PATH, "simulate/logs/")+"-split-{}.csv".format(i) for i in range(PARALLEL_WORKER)]
road_info_files = [ROAD_INFO_FILE+"-split-{}.csv".format(i) for i in range(PARALLEL_WORKER)]

def task_partition_csv(input_file=AOI_FILE, input_files=input_aoi_files):
    tasks = pd.read_csv(input_file)
    split_tasks = np.array_split(tasks, PARALLEL_WORKER)
    for i, task in enumerate(split_tasks):
        task.to_csv(input_files[i])

task_partition_csv(AOI_FILE, input_aoi_files)

# 准备参数
commands = []
for i in range(PARALLEL_WORKER):
    command = ["python", "-m", "simulate.train_task"]
    command.append("--simulate_input_file")
    command.append(output_files[i])
    command.append("--road_info_file")
    command.append(road_info_files[i])
    command.append("--aoi_file")
    command.append(input_aoi_files[i])
    command.append("--parallel")
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
for file in output_files+input_aoi_files:
    os.remove(file)

