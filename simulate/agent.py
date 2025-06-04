import os
import openai
import copy
import time
import json
import signal
import asyncio
import argparse
import jsonlines
from typing import List
from aiomultiprocess import Pool

from pycitydata.map import Map
from citysim.routing import RoutingClient

from simulate.player import TextPlayer
from simulate.utils import RunMode, Action, lnglat2grid, load_map, process_action, simplify_observation_pois
from simulate.prompts import get_prompts, get_system_instruction, get_available_actions
from simulate.templates import init_prompt_text
from config import REGION_EXP, DETAIL_INTEREST, VISION_DATA, MAP_CACHE_PATH, ROUTING_PATH, MAP_DICT, MAP_PORT_DICT, MIN_ROAD_LENGTH, MONGODB_URI

def inject_info(session: List, history: List):
    current_role = "user"
    traverse = {"user": "assistant", "assistant": "user"}
    for his in history:
        session.append({"role": current_role, "content": his})
        current_role = traverse[current_role]
    return session



async def main(args):
    samples = args.samples
    model_name = args.model_name
    continue_record = args.continue_record==1
    run_mode = args.run_mode
    nearby_params = args.nearby_params
    input_file_name = args.input_file
    output_file_name = args.output_file
    workers = args.workers

    city_map = MAP_DICT[REGION_EXP]
    port = args.port if args.port is not None else MAP_PORT_DICT[REGION_EXP]
    if workers == True:
        m = Map(
            mongo_uri=MONGODB_URI,
            mongo_db="llmsim",
            mongo_coll=city_map,
            cache_dir=MAP_CACHE_PATH,
        )
        routing_client = RoutingClient("localhost:{}".format(port))
    else:
        m, process, routing_client = load_map(
            city_map=city_map, 
            cache_dir=MAP_CACHE_PATH, 
            routing_path=ROUTING_PATH, 
            port=port)
        await asyncio.sleep(20)

    # 限制最短路径长度
    min_road_length = MIN_ROAD_LENGTH

    if REGION_EXP == "example":
        last_time = time.time()
        task_info = {
            "goal": "松下电器(红星美凯龙北四环店)", 
            "poi_id": 700907209, 
            "init_aoi": 500116476, 
            "init_poi": 700559061, 
            "task": "You are in 北太平庄中路甲43号院 and you need to go to 松下电器(红星美凯龙北四环店). Your current position is longitude:116.3660 latitude:39.9715.",
            "region": "wudaokou", 
            "type": "citywalk"
            }
        done, log_info, final_reason = await single_run_city_walk(task_info=task_info, m=m, routing_client=routing_client, model_name=model_name, min_road_length=min_road_length)

        res_info = copy.deepcopy(task_info)
        res_info["done"] = done
        res_info["log_info"] = log_info
        res_info["time_cost"] = time.time()-last_time
        res_info["final_reason"] = final_reason

        with open("simulate/logs/examples_{}.json".format(model_name), "w", encoding="utf-8") as wid:
            json.dump(res_info, wid, ensure_ascii=False, indent=2)
        with jsonlines.open("simulate/logs/example_{}.jsonl".format(model_name), "w") as wid:
            wid.write(res_info)
    else:
        tasks = []

        # 读取任务数据
        with jsonlines.open(input_file_name) as fid:
            for i, task_info in enumerate(fid):
                tasks.append(task_info)

        # 读取已经存在的输出文件，方便续写
        res = []
        if continue_record:
            try:
                with jsonlines.open(output_file_name) as fid:
                    for obj in fid:
                        res.append(obj)
            except FileNotFoundError as e:
                print("没有历史记录，直接重头开始. {}".format(e))
        
        start_time = time.time()
        # 开始执行任务并记录结果
        with jsonlines.open(output_file_name, "a" if continue_record else "w") as wid:
            for i, task_info in enumerate(tasks):
                # 跳过已经跑过的数据
                if i<len(res):
                    continue
                
                # 只采样部分数据进行测试
                if i==samples:
                    break

                last_time = time.time()
                
                done, log_info, final_reason = await single_run_city_walk(task_info=task_info, m=m, routing_client=routing_client, model_name=model_name, to_print=False, min_road_length=min_road_length, nearby_params=nearby_params)
                res_info = copy.deepcopy(task_info)

                res_info["done"] = done
                res_info["log_info"] = log_info
                res_info["time_cost"] = time.time()-last_time
                res_info["final_reason"] = final_reason

                res.append(res_info)
                i+=1

                wid.write(res_info)

                if i%100==99:
                    print("complete {}/{} time_cost:{}".format(i, len(tasks), time.time()-start_time))

        # json格式便于阅读分析，jsonl方便共享训练数据
        with open(output_file_name.replace("jsonl", "json"), "w", encoding="utf-8") as wid:
            json.dump(res, wid, ensure_ascii=False, indent=2)

    if workers == False:
        print("send signal")
        process.send_signal(sig=signal.SIGTERM)
        process.wait()


async def single_run_city_walk(task_info: dict, m: Map, routing_client: RoutingClient, model_name: str, to_print=True, min_road_length=100, nearby_params={"radius": 100, "limit": 10, "has_category": True}):
    init_aoi_id = task_info["init_aoi"]
    region_exp = task_info["region"]
    init_poi_id = task_info["init_poi"]
    current_task = {"task": task_info["task"], "goal": task_info["goal"], "init_poi": task_info["init_poi"]}
    env = TextPlayer(m, routing_client, init_aoi_id, min_road_length, region_exp, nearby_params=nearby_params, init_poi_id=init_poi_id)
    env.reset()
    env.set_max_episode_length(20)
    env.register_poi_info(poi_name=task_info["goal"], poi_id=task_info["poi_id"])

    log_info = {"log": []}

    session = []
    session.append({"role": "user", "content": get_system_instruction(run_mode=RunMode.CITY_WALK.value)})
    session.append({"role": "assistant", "content": "OK. I'll follow your instructions and try my best to solve the task."})

    available_actions_text = get_available_actions(env.get_action_space(run_mode=RunMode.CITY_WALK.value, goal=current_task["goal"]))
    init_prompt = init_prompt_text(current_task, available_actions_text)

    log_info["init_prompt"] = init_prompt
    session.append({"role": "user", "content": init_prompt})
    if to_print:
        print("###############Start Task###############")
        print(init_prompt)

    final_reason = ""
    done = False
    # last_time = time.time()
    for i in range(0, env.max_episode_length):
        if i==0:
            output = "ACTION: {} {}.".format(Action.NAVIGATE.value, current_task["goal"])
        else:
            output = "ACTION: {} {}".format(Action.WALK.value, current_task["goal"])
        
        if i==0:
            session.append({"role":"assistant", "content": "start from {} ".format(current_task["init_poi"])+output})
        else:
            session.append({"role":"assistant", "content": output})

        available_actions = env.get_action_space(run_mode=RunMode.CITY_WALK.value, goal=current_task["goal"])
        action_object = process_action(output, available_actions)

        # 测试时去除try-except
        observation, reward, done, info = await env.step(action_object, run_mode=RunMode.CITY_WALK.value, detail_interest=DETAIL_INTEREST)   
        
        # try:
        #     observation, reward, done, info = await env.step(action_object, run_mode=RunMode.CITY_WALK.value, detail_interest=DETAIL_INTEREST)
        # except NotImplementedError as e:
        #     done= False
        #     final_reason = SampleStatus.AGENT_INVALID_ACTION.value
        #     session.append({"role": "user", "content": final_reason + "thus nothing happens."})
        #     continue
        # except Exception as e:
        #     done = False
        #     final_reason = SampleStatus.UNKNOWN.value+" "+str(e)
        #     break
        
        
        # 将环境反馈信息以user视角加入对话历史
        session.append({
                "role": "user", 
                # "content": observation["observations_text"] + get_available_actions(available_actions)
                "content": observation["observations_text"]
                })
        if VISION_DATA == True:
            files = observation["files"]
            if len(files) != 0:
                files_content = ", ".join(files)
                session.append({"role": "image_paths", "content": files_content})

        # save
        payload = {
            "round": i + 1,
            "output": output,
            "action": action_object,
            "observation": observation["observations_text"],
            "observation_origin": simplify_observation_pois(m, observation, DETAIL_INTEREST), # 为了后续评估任务构建，记录中间数据，有些模式下并不一定具备
            "done": done,
        }
        log_info["log"].append(payload)
        
        if to_print:
            print("##############################")
            print("round:  ", i+1)
            print("output: ", output)
            print("action: ", action_object)
            # print("available actions: ", available_actions)
            print("history obs and commands: ", session[-2]["content"].replace("\n", ";"))
            print("#############")
            print("current ob (text format): ", observation["observations_text"])
            # print("ob json: ", observation["observations"])
            print("##############################")


        # 任务是否结束判断
        if done:
            final_reason = info["reason"]
            break
    else:
        final_reason = "task limit reached"

    # 记录最后形成的完整对话
    log_info["session"] =  session

    if to_print:
        if len(log_info["log"])>0 and log_info["log"][-1]["done"]:
            print("Task <{}> is completed.".format(current_task["task"]))
        else:
            print("Task <{}> failed with reason:{}.".format(current_task["task"], final_reason))
        
    return done, log_info, final_reason


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="mock")
    parser.add_argument("--continue_record", type=int, default=0, choices=[1, 0], help="数据记录时是否保留已有数据")
    parser.add_argument("--run_mode", default=RunMode.CITY_WALK.value, type=str, choices=[RunMode.NORMAL.value, RunMode.CITY_WALK.value])
    parser.add_argument("--nearby_radius", default=100, type=int)
    parser.add_argument("--nearby_limit", default=10, type=int)
    parser.add_argument("--nearby_has_category", default=1, type=int)
    parser.add_argument("--input_file", default="simulate/tasks/input_citywalk_wudaokou_small-v10.3-english-chinese-test.jsonl")
    parser.add_argument("--output_file", default="simulate/logs/output_citywalk_wudaokou-english-chinese-test.jsonl")
    parser.add_argument("--workers", action="store_true", help="是否使用多进程")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()


    args.nearby_params = {"radius": args.nearby_radius, "limit": args.nearby_limit, "has_category": args.nearby_has_category}
    print(args.nearby_params)


    # 简单串行
    asyncio.run(main(args))
