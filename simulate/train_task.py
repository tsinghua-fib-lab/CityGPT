import os
import argparse
import numpy as np
import pandas as pd
import random
import jsonlines
import json
import signal
import asyncio
from collections import Counter

from tqdm import tqdm
from pycitydata.map import Map
from citysim.routing import RoutingClient

random.seed(42)

from config import REGION_EXP, DATA_VERSION, RESOURCE_PATH, MAP_CACHE_PATH, ROUTING_PATH, MAP_DICT, MAP_PORT_DICT, MONGODB_URI
from simulate.translate import Name
LANGUAGE = Name(region_exp=REGION_EXP)
from simulate.player import Player
from simulate.templates import task_description_text
from simulate.utils import load_map

# 定义得到的文件中数据路径是否去重, 参数为True时去重，False时不去重
DEDUPLICATION = True

class RoadNavigate(Player):
    def __init__(
        self,
        city_map: Map,
        city_routing_client: RoutingClient,
        init_aoi_id: int,
        road_info_file:str,
        init_poi_id = None,
        nearby_params={"radius": 100, "limit": 10, "has_category": 0}
    ):
        """
        类比virtual-home: region->home, aoi->room, poi->object, service->action
        """
        self.init_aoi_id = init_aoi_id
        if init_poi_id is not None:
            self.init_poi_id = init_poi_id
        self.stored_routes = set()
        self.road_info_file = road_info_file
        super().__init__(
            city_map=city_map,
            city_routing_client=city_routing_client,
            init_aoi_id=init_aoi_id
        )

async def check_route(start_aoi_id, dest_aoi_id, existing_routes, road_info_file, map, routing_client, start_poi_id=None):
    enva = RoadNavigate(city_map=map, city_routing_client=routing_client, init_aoi_id=start_aoi_id, road_info_file=road_info_file, init_poi_id=start_poi_id)
    route = await enva.get_driving_route(dest_aoi_id)
    if route is None:
        return None
    
    if len(route["road_ids"]) > 10:
        return None
    
    road_ids_str = json.dumps(route["road_ids"])
    if road_ids_str in existing_routes:
        return None
    else:
        existing_routes.add(road_ids_str)
        return road_ids_str

async def generate_tasks_citywalk(
    simulate_input_file,
    road_info_file,
    aoi_file,
    map=None,
    routing_client=None):
    min_pois = 2
    name2pois = LANGUAGE.pois_data.set_index("name")["poi_id"].to_dict()
    candidate_names = list(name2pois.keys())

    aois_data = pd.read_csv(aoi_file)
    aois_data_slim = []
    min_pois = 1
    for aoi_id in aois_data.aoi_id.to_list():
        info = map.get_aoi(id=aoi_id)
        if len(info["poi_ids"]) < min_pois:
            continue
        aois_data_slim.append(info)
    random.shuffle(aois_data_slim)

    aois_data_all = LANGUAGE.aois_data
    aois_data_all_slim = []
    for aoi_id in aois_data_all.aoi_id.to_list():
        info = map.get_aoi(id=aoi_id)
        if len(info["poi_ids"]) < min_pois:
            continue
        aois_data_all_slim.append(info)
    random.shuffle(aois_data_all_slim)

    existing_routes = set()
    train_task = []
    for start_aoi in aois_data_slim:
        # start aoi从切分aoi文件中选
        start_poi_ids = [poi_id for poi_id in start_aoi["poi_ids"] if LANGUAGE.get_poi_name(poi_id, map) != "" and pd.notna(LANGUAGE.get_poi_name(poi_id, map))]
        if len(start_poi_ids) == 0:
            continue

        # 随机选择 k 个与 start_aoi 不同的 dest_aoi，从all aoi中选
        dest_aois = [aoi for aoi in aois_data_all_slim if aoi["id"] != start_aoi["id"]]
        random.shuffle(dest_aois)
        dest_aois = dest_aois[:500]  # 限制目标 AOI 数量
        
        for dest_aoi in dest_aois:
            # 从 start_aoi 中随机选择一个 POI 作为起点
            start_poi_id = random.choice(start_poi_ids)
            start_poi_name = LANGUAGE.get_poi_name(start_poi_id, map)
            start_poi_addr = LANGUAGE.get_poi_address(start_poi_id)
            if start_poi_addr == "":
                continue
            else:
                start_poi_addr_str = f"({start_poi_addr})"
            
            # 从 dest_aoi 中选择一个 POI 作为终点
            dest_poi_ids = [poi_id for poi_id in dest_aoi["poi_ids"] if LANGUAGE.get_poi_name(poi_id, map) != "" and pd.notna(LANGUAGE.get_poi_name(poi_id, map))]
            if len(dest_poi_ids) == 0:
                continue
            
            dest_poi_id = random.choice(dest_poi_ids)
            dest_poi_name = LANGUAGE.get_poi_name(dest_poi_id, map)
            dest_poi_addr = LANGUAGE.get_poi_address(dest_poi_id)
            if dest_poi_addr == "":
                continue
            else:
                dest_poi_addr_str = "({})".format(dest_poi_addr)
            coords = map.get_poi(start_poi_id)["position"]
            lng, lat = map.xy2lnglat(x=coords["x"], y=coords["y"])
            
            task_text = task_description_text(start_poi_name, start_poi_addr_str, dest_poi_name, dest_poi_addr_str, lng, lat)

            if DEDUPLICATION:
                result = await check_route(start_aoi["id"], dest_aoi["id"], existing_routes, road_info_file, map, routing_client, start_poi_id)
                if result is None:
                    # if start_aoi["id"] != dest_aoi["id"]:
                        # print("start_aoi:{} dest_aoi:{} same route".format(start_aoi["id"], dest_aoi["id"]))
                    # print("route duplicated")
                    continue
            else:
                result = None
            train_task.append(
                {
                    "goal": dest_poi_name,
                    "poi_id": dest_poi_id,
                    "init_aoi": start_aoi["id"],
                    "init_poi": start_poi_id,
                    "task": task_text,
                    "routing_list": result,
                    "region": REGION_EXP,
                    "type": "citywalk"
                } )

    if len(train_task)>0:
        with jsonlines.open(simulate_input_file, "w") as train_wid:
            for task in train_task:
                train_wid.write(task)
        print(f"Tasks saved to {simulate_input_file}")


async def main(args):
    city_map = MAP_DICT[REGION_EXP]
    # port = args.port if args.port is not None else MAP_PORT_DICT[REGION_EXP]
    port = MAP_PORT_DICT[REGION_EXP]
    if args.parallel:
        map = Map(
                mongo_uri=f"{MONGODB_URI}",
                mongo_db="srt",
                mongo_coll=city_map,
                cache_dir=MAP_CACHE_PATH,
        )
        routing_client = RoutingClient(f"localhost:{port}")
    else:
        map, process, routing_client = load_map(
            city_map=city_map, 
            cache_dir=MAP_CACHE_PATH, 
            routing_path=ROUTING_PATH, 
            port=port)

    # 原始simulate文件名
    # simulate_input_file = "simulate/tasks/input_citywalk_{}-{}.jsonl".format(REGION_EXP, DATA_VERSION)
    # # 生成路径中经过的road_id信息文件名
    # road_info_file = "simulate/logs/roads_info_{}-{}.csv".format(REGION_EXP, DATA_VERSION)
    # # 初始化时重置文件，防止一直累加数据
    # with open(road_info_file, "w") as wid:
    #     pass
    print("start building dataset for {}-{}".format(REGION_EXP, DATA_VERSION))
    # 生成基础训练数据
    await (generate_tasks_citywalk(
        args.simulate_input_file, 
        args.road_info_file, 
        args.aoi_file, 
        map=map, 
        routing_client=routing_client))
    if not args.parallel:
        print("send signal")
        process.send_signal(sig=signal.SIGTERM)
        process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate_input_file", type=str, required=True)
    parser.add_argument("--road_info_file", type=str, required=True)
    parser.add_argument("--aoi_file", type=str, required=True)  # 传入切分后的 AOI 文件
    parser.add_argument("--parallel", action="store_true", help="控制是否并行执行")
    args = parser.parse_args()
    
    asyncio.run(main(args))

