import os
import tqdm
import time
import numpy as np
from typing import List, Tuple
from multiprocessing import Pool

from moss import Engine, LaneChange, TlPolicy

from serving.llm_api import get_response_traffic_signal
from config import LLM_MODEL_MAPPING
from .utils import JunctionState, validate_response, process_phase, get_chosen_number_from_phase_index, whether_junc_in_region, get_coords

# 基于LLM输出进行交通灯控制
def post_process(response, phase_map, qualified_phase_num, j, state, eng):
    answer = validate_response(response, phase_map)
    
    if answer[0] == True:
        chosen_number = answer[1]
        state.phase_index = phase_map[chosen_number]
        # print(f"Chosen number: {chosen_number}")
        # print(f"Chosen phase index: {state.phase_index}")
    else:
        # print(answer[1])
        # print(state.phase_index)
        # print(j.index)
        phase_index = state.phase_index if state.phase_index>0 else list(phase_map.values())[0]
        chosen_number = get_chosen_number_from_phase_index(phase_index, phase_map)
        
        new_chosen_number = (chosen_number + 1) % qualified_phase_num
        state.phase_index = phase_map[new_chosen_number]
        # print(f"Chosen number: {chosen_number}")
        # print(f"New Chosen number: {new_chosen_number}")
        # print(f"Chosen phase index: {state.phase_index}")
    
    # 设置路口相位
    # print(f"j.index: {j.index}, junction_index: {junction_index}")
    eng.set_tl_phase(j.index, state.phase_index)
    # 返回处理的交叉口索引
    return answer[0]


# 基于模拟器状态构建LLM的prompt
def get_prompt(j, cnt, waiting_cnt):
    # 保存相位信息
    phase_options = []
    # 相位选项序号映射
    phase_map = {} 
    # 过滤掉黄灯后的相位选项序号
    filtered_index = 0
    
    for original_index, phase in enumerate(j.tl.phases):
        phase_valid = True
        green_indices = []

        for i, phase_state in enumerate(phase.states):
            if phase_state.name == "YELLOW":
                # 过滤掉黄灯选项
                phase_valid = False
                break
            if phase_state.name == "GREEN":
                # 记录绿灯序号
                green_indices.append(i)

        if phase_valid:
            # 相位序号做映射
            phase_map[filtered_index] = original_index
            description = process_phase_prompt(j, green_indices, filtered_index, cnt, waiting_cnt)
            filtered_index += 1
            phase_options.append(description)

    # 过滤掉黄灯的总相位数
    qualified_phase_num = filtered_index
    # 合并车道信息和相位选项信息
    basic_descriptions = f"""A traffic light controls a complex intersection with {qualified_phase_num} signal phases. Each signal relieves vehicles' flow in the allowed lanes.
    The state of the intersection is listed below. It describes:
    - The number of lanes relieving vehicles' flow under each traffic light phase.
    - The total count of vehicles present on lanes permitted to move of each signal.
    - The count of vehicles queuing on lanes allowed to move, which are close to the end of the road or intersection and moving at low speeds of each signal."""
    prompt = "\n".join([basic_descriptions] + ["Available Phase Options"] + phase_options)
    
    prompt += """
    Please answer:
    Which is the most effective traffic signal that will most significantly improve the traffic condition during the
    next phase?

    Note:
    The traffic congestion is primarily dictated by the waiting vehicles, with the MOST significant impact. You MUST pay the MOST attention to lanes with long queue lengths.

    Requirements:
    - Let's think step by step.
    - You can only choose one of the phase option NUMBER listed above.
    - You must follow the following steps to provide your analysis:
        Step 1: Provide your analysis for identifying the optimal traffic signal.
        Step 2: Answer your chosen phase option number.
    - Your choice can only be given after finishing the analysis.
    - Your choice must be identified by the tag: <signal>YOUR_NUMBER</signal>.

    """
    
    return prompt, phase_map, qualified_phase_num


def process_phase_prompt(j, green_indices, filtered_index, cnt, waiting_cnt):
    filtered_index, lane_counts, vehicle_counts, waiting_vehicle_counts = process_phase(j, green_indices, filtered_index, cnt, waiting_cnt)
    description = f"Phase Option {filtered_index}:\n -Allowed lane counts: {lane_counts},\n -Vehicle counts: {vehicle_counts},\n -Waiting vehicle counts: {waiting_vehicle_counts}"
    return description



#################
def simulate(CITY, MAP_FILE, AGENT_FILE, TOTAL_TIME, START_TIME, MODEL_NAME):
    TIME_THRESHOLD = 30
    GPU_DEVICE_NUM = 0

    eng = Engine(
        map_file=MAP_FILE,
        agent_file= AGENT_FILE,
        start_step=START_TIME,
        lane_change=LaneChange.MOBIL,
        device=GPU_DEVICE_NUM
    )
    if MODEL_NAME == "max-pressure":
        # 设置所有路口的信控为MAX_PRESSURE
        eng.set_tl_policy_batch(range(eng.junction_count), TlPolicy.MAX_PRESSURE)
    else:
        # 设置所有路口的信控为FIXED_TIME
        eng.set_tl_policy_batch(range(eng.junction_count), TlPolicy.FIXED_TIME)
    eng.set_tl_duration_batch(range(eng.junction_count), TIME_THRESHOLD)

    model = None
    if MODEL_NAME == "fixed-time" or MODEL_NAME == "max-pressure":
        model_full = MODEL_NAME
    else:
        try:
            model_full = LLM_MODEL_MAPPING[MODEL_NAME]
        except KeyError:
            raise NotImplementedError
            # model_full = MODEL_NAME
            # model_wrapper = VLMWrapper(model_full)
            # model = model_wrapper.get_vlm_model()


    # 获取地图
    M = eng.get_map()
    # 划分的交通灯控制区域
    coords = get_coords(CITY)
    # 保存需要的交叉口索引
    valid_junctions = []
    junc_states = {}
    for j in M.junctions:
        # j = M.junctions[junction_index]
        if j.tl is None:
            continue
        if whether_junc_in_region(M, j, coords) == False:
            continue
        # 设置指定路口的信控为手动切换
        if MODEL_NAME=="fixed-time":
            pass
        elif MODEL_NAME=="max-pressure":
            pass
        else:
            eng.set_tl_policy(j.index, TlPolicy.MANUAL)
        junc_states[j.index] = JunctionState()
        valid_junctions.append(j.index)

    print(f"valid_junctions: {len(valid_junctions)}")

    aql = []

    print("Start Simulate")
    sim_time = time.time()
    llm_cost_time_sum = 0
    invalid_actions = 0
    llm_inout = []
    for epoch in tqdm.tqdm(range(TOTAL_TIME//TIME_THRESHOLD)):
        # 获取lane上所有的车辆数
        cnt = eng.get_lane_vehicle_counts()  
        waiting_cnt = eng.get_lane_waiting_at_end_vehicle_counts() 
        # Average queue length
        print(f"current vehicles sum:{cnt.sum()} waiting vehicles sum:{waiting_cnt.sum()}")
        aql.append(waiting_cnt.sum()/len(valid_junctions))
        
        junction_info = []
        junction_prompt = []
        # 获取所有junction参数
        for junction_index in valid_junctions:
            j = M.junctions[junction_index]
            state = junc_states[j.index]
            prompt, phase_map, qualified_phase_num = get_prompt(j, cnt, waiting_cnt)
            junction_info.append([prompt, phase_map, qualified_phase_num, j, state])
            junction_prompt.append([prompt, model_full, 500, 0]) # ,model
        
        # 并行处理交叉路口LLM访问请求
        llm_start_time = time.time()
        with Pool(processes=len(valid_junctions)) as pool:
            results = pool.starmap(get_response_traffic_signal, junction_prompt)
        llm_cost_time = time.time()-llm_start_time
        llm_cost_time_sum += llm_cost_time
        
        # 基于LLM回答进行后处理
        for i, res in enumerate(results):
            if MODEL_NAME == "fixed-time":
                pass
            elif MODEL_NAME == "max-pressure":
                pass
            else:
                llm_inout.append([{"role":"user", "content": junction_prompt[i][0]}, {"role":"assistant", "content": res}])
                ans_state = post_process(res, junction_info[i][1], junction_info[i][2], junction_info[i][3], junction_info[i][4], eng)
                if not ans_state:
                    invalid_actions += 1

        print("Steps: {} Method: {} Cost: {}".format(epoch, MODEL_NAME, llm_cost_time))
        eng.next_step(TIME_THRESHOLD)

    ####### 计算指标
    sim_time = time.time()-sim_time
    # Average queue length
    aql = np.mean(aql)
    # Average traveling time
    att = eng.get_departed_vehicle_average_traveling_time()
    # Throughput
    tp = eng.get_finished_vehicle_count()

    return aql, att, tp, sim_time, llm_cost_time_sum, invalid_actions, llm_inout
