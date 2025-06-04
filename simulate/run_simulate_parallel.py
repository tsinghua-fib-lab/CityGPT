import os
import argparse
import jsonlines
import subprocess
from simulate.utils import RunMode
from config import REGION_EXP, DATA_VERSION

PARALLEL_WORKER=20 # 并行模拟数量

### 基础参数
SAMPLES=210000          # 预期生成的样本数量，实际生成样本数为 min(SAMPLES,INPUT_FILE中的样本数)
MODEL_NAME="mock"   # 实际执行citywalk的agent背后的支撑模型，mock意味着最简单的规则模型
CONTINUE_RECORD=0   # 是否需要保留之前样本，接着生成
RUN_MODE="citywalk" # 目前基本保持不变即可

### 指定citywalk相关参数
radius=100          # 当前位置半径xx米
limit=10            # 当前位置查询POI数量限制
has_category=1      # 查询信息是否包含category，1代表包含
file_name_nearby="{}_{}_{}".format(radius, limit, has_category)

### 定义 输入、输出文件名
INPUT_FILE="simulate/tasks/input_citywalk_{}-{}.jsonl".format(REGION_EXP, DATA_VERSION)
OUTPUT_FILE="simulate/logs/output_{}_{}_{}_{}_{}.jsonl".format(RUN_MODE, REGION_EXP, MODEL_NAME, file_name_nearby, DATA_VERSION)
SFT_FILE="simulate/examples/citywalk-{}-{}-{}.jsonl".format(REGION_EXP, MODEL_NAME, DATA_VERSION)


# 可以通过命令行重置参数
parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=SAMPLES)
parser.add_argument("--model_name", type=str, default=MODEL_NAME)
parser.add_argument("--continue_record", type=int, default=CONTINUE_RECORD, choices=[1, 0], help="数据记录时是否保留已有数据")
parser.add_argument("--run_mode", default=RUN_MODE, type=str, choices=[RunMode.NORMAL.value, RunMode.CITY_WALK.value])
parser.add_argument("--nearby_radius", default=radius, type=int)
parser.add_argument("--nearby_limit", default=limit, type=int)
parser.add_argument("--nearby_has_category", default=has_category, type=int)
parser.add_argument("--input_file", default=INPUT_FILE)
parser.add_argument("--output_file", default=OUTPUT_FILE)
args = parser.parse_args()


# 切割文件  
tasks = []
with jsonlines.open(args.input_file) as fid:
    for i, task_info in enumerate(fid):
        tasks.append(task_info)
chunk_size, remainder = len(tasks) // PARALLEL_WORKER, len(tasks) % PARALLEL_WORKER
tasks_parallel = [tasks[i*chunk_size:(i+1)*chunk_size] for i in range(PARALLEL_WORKER)]
if remainder>0:
    tasks_parallel[-1] += tasks[-remainder:]

input_files = [args.input_file.replace("tasks/", "logs/")+"-split-{}.jsonl".format(i) for i in range(PARALLEL_WORKER)]
output_files = [args.output_file+"-split-{}.jsonl".format(i) for i in range(PARALLEL_WORKER)]
for i, input_file in enumerate(input_files):
    with jsonlines.open(input_file, "w") as wid:
        wid.write_all(tasks_parallel[i])

# 准备参数
commands = []
for i in range(PARALLEL_WORKER):
    command = ["python", "-m", "simulate.agent"]
    args_dict = args.__dict__
    for x in args_dict:
        command.append("--{}".format(x))
        if x in "input_file":
            command.append(input_files[i])
        elif x in "output_file":
            command.append(output_files[i])
        else:
            command.append(str(args_dict[x]))
    command.append("--workers")
    commands.append(command)

# 并行执行
processes = []
for i, c in enumerate(commands):
    if len(commands)<=5:
        print("start task:{} command:{}".format(i, c))
    processes.append(subprocess.Popen(c))
for process in processes:
    process.wait()
print("所有进程执行完成,开始整合文件")

# 合并文件
with jsonlines.open(args.output_file, "w") as wid:
    for output_file in output_files:
        with jsonlines.open(output_file, "r") as fid:
            for item in fid:
                wid.write(item)

# 后处理代码
subprocess.run(["python", "-m", "simulate.process", "--log_file", OUTPUT_FILE, "--sft_file", SFT_FILE])

common_path = "/data1/citygpt/datasets" # 下面有citywalk，merge，general，cache四个路径
os.popen("cp {} {}".format(SFT_FILE, os.path.join(common_path, "citywalk")))

# 删除临时文件
for file in input_files:
    os.remove(file)
for file in output_files:
    os.remove(file)
for file in output_files:
    os.remove(file.replace(".jsonl", ".json"))
