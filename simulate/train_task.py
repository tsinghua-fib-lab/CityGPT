import os
import argparse
import numpy as np
from pycitysim.map import Map
import pandas as pd
import random
import jsonlines
import json
import time
import signal
import asyncio
from collections import Counter
from pycitysim.routing import RoutingClient

random.seed(42)

from config import REGION_EXP, DATA_VERSION, RESOURCE_PATH, MAP_CACHE_PATH, ROUTING_PATH, MAP_DICT, MAP_PORT_DICT, OSM_REGION
from simulate.translate import Name
LANGUAGE = Name(region_exp=REGION_EXP)
from simulate.player import Player
from simulate.templates import task_description_text
from simulate.utils import load_map
from simulate.address_system import get_center_point

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
    if OSM_REGION == True:
        enva = RoadNavigate(city_map=map, city_routing_client=routing_client, init_aoi_id=start_aoi_id, road_info_file=road_info_file)
    else:
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
    region_exp="wudaokou",
    data_version=None,
    map=None,
    routing_client=None):
    
    if OSM_REGION == True:
        name2aois = LANGUAGE.aois_data.set_index("aoi_name")["aoi_id"].to_dict()
        candidate_names = list(name2aois.keys())
    else:
        name2pois = LANGUAGE.pois_data.set_index("name")["poi_id"].to_dict()
        candidate_names = list(name2pois.keys())

    aois_data_slim = []
    min_pois = 1
    for aoi_id in LANGUAGE.aois_data.aoi_id.to_list():
        info = map.get_aoi(id=aoi_id)
        if OSM_REGION == False:
            if len(info["poi_ids"]) < min_pois:
                continue
        aois_data_slim.append(info)
    
    random.shuffle(aois_data_slim)
    existing_routes = set()
    train_task = []
    for aoi in aois_data_slim:
        if OSM_REGION == True:
            start_aoi_id = aoi["id"]
            start_aoi_name = LANGUAGE.get_aoi_name(start_aoi_id,map)
            # 过滤掉与 start_aoi_name 相同的名称
            filtered_candidates = [name for name in candidate_names if name != start_aoi_name]
        
        # 保证有足够的候选名称供随机抽样
            for dest_aoi_name in random.sample(filtered_candidates, k=3):
                # 从出发AOI中随机选择POI, 并记录其位置
                dest_aoi_id = name2aois[dest_aoi_name]
                start_aoi_addr = LANGUAGE.get_aoi_address(start_aoi_id)
                if start_aoi_addr == "":
                    start_aoi_addr_str = ""
                else:
                    start_aoi_addr_str = "({})".format(start_aoi_addr)
                dest_aoi_addr = LANGUAGE.get_aoi_address(dest_aoi_id)
                if dest_aoi_addr == "":
                    dest_aoi_addr_str = ""
                else:
                    dest_aoi_addr_str = "({})".format(dest_aoi_addr)
                coords = get_center_point(map.get_aoi(start_aoi_id)['shapely_xy'])
                lng, lat = map.xy2lnglat(x=coords.x, y=coords.y)
                # task_text = task_description_text(start_aoi_name, start_aoi_addr_str, dest_aoi_name, dest_aoi_addr_str, lng, lat, TEXT_ENGLISH)
                task_text = "You are in {}{} and you need to go to {}{}. Your current position is longitude:{:.4f} latitude:{:.4f}.".format(start_aoi_name, start_aoi_addr_str, dest_aoi_name, dest_aoi_addr_str, lng, lat)

                if DEDUPLICATION:
                    result = await check_route(aoi["id"], dest_aoi_id, existing_routes, road_info_file, map, routing_client)
                    if result is None:
                        continue
                else:
                    result = None
                train_task.append(
                    {
                        "goal": dest_aoi_name,
                        "aoi_id": dest_aoi_id,
                        "init_aoi": aoi["id"],
                        "task": task_text,
                        "routing_list": result,
                        "region": region_exp,
                        "type": "citywalk"
                    } )
        else:
            include_poi_ids = [poi_id for poi_id in aoi["poi_ids"] if LANGUAGE.get_poi_name(poi_id, map) != "" and pd.notna(LANGUAGE.get_poi_name(poi_id, map))]
            if len(include_poi_ids)==0:
                continue
            
            # 确保选择的dest_poi与start_poi不属于同一aoi，且poi_name不为空
            filtered_candidate_names = [name for name in candidate_names if not pd.isna(name) and name != ""  and map.get_poi(name2pois[name])["aoi_id"] != aoi["id"]]
            for dest_poi_name in random.sample(filtered_candidate_names, k=3):
            # print(f"length of filtered_candidate_names: {len(filtered_candidate_names)}")
                # 从出发AOI中随机选择POI, 并记录其位置
                start_poi_id = random.choice(include_poi_ids)
                start_poi_name = LANGUAGE.get_poi_name(start_poi_id, map)
                if start_poi_name == "":
                    print("start_poi_name is empty")
                    continue
                dest_poi_id = name2pois[dest_poi_name]
                dest_aoi_id = map.get_poi(dest_poi_id)["aoi_id"]
                start_poi_addr = LANGUAGE.get_poi_address(start_poi_id)
                if start_poi_addr == "":
                    start_poi_addr_str = ""
                else:
                    start_poi_addr_str = "({})".format(start_poi_addr)
                dest_poi_addr = LANGUAGE.get_poi_address(dest_poi_id)
                if dest_poi_addr == "":
                    dest_poi_addr_str = ""
                else:
                    dest_poi_addr_str = "({})".format(dest_poi_addr)
                coords = map.get_poi(start_poi_id)["position"]
                lng, lat = map.xy2lnglat(x=coords["x"], y=coords["y"])
                
                task_text = task_description_text(start_poi_name, start_poi_addr_str, dest_poi_name, dest_poi_addr_str, lng, lat)

                if DEDUPLICATION:
                    result = await check_route(aoi["id"], dest_aoi_id, existing_routes, road_info_file, map, routing_client, start_poi_id)
                    if result is None:
                        continue
                else:
                    result = None
                train_task.append(
                    {
                        "goal": dest_poi_name,
                        "poi_id": dest_poi_id,
                        "init_aoi": aoi["id"],
                        "init_poi": start_poi_id,
                        "task": task_text,
                        "routing_list": result,
                        "region": region_exp,
                        "type": "citywalk"
                    } )

    if len(train_task)>0:
        with jsonlines.open(simulate_input_file, "w") as train_wid:
            for task in train_task:
                train_wid.write(task)


async def main(args):
    city_map = MAP_DICT[REGION_EXP]
    port = args.port if args.port is not None else MAP_PORT_DICT[REGION_EXP]
    map, process, routing_client = load_map(
        city_map=city_map, 
        cache_dir=MAP_CACHE_PATH, 
        routing_path=ROUTING_PATH, 
        port=port)
    time.sleep(10)

    # 原始simulate文件名
    simulate_input_file = "simulate/tasks/input_citywalk_{}-{}.jsonl".format(args.region, args.data_version)
    # 生成路径中经过的road_id信息文件名
    road_info_file = "simulate/logs/roads_info_{}-{}.csv".format(args.region, args.data_version)
    # 初始化时重置文件，防止一直累加数据
    with open(road_info_file, "w") as wid:
        pass

    print("start building dataset for {}-{}".format(args.region, args.data_version))
    # 生成基础训练数据
    await (generate_tasks_citywalk(
        simulate_input_file, 
        road_info_file, 
        region_exp=args.region, 
        data_version=args.data_version, 
        map=map, 
        routing_client=routing_client))
    
    print("send signal")
    process.send_signal(sig=signal.SIGTERM)
    process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=REGION_EXP, choices=["example", "beijing", "wudaokou_small", "wudaokou_large", "wangjing", "dahongmen", "paris", "newyork"])
    parser.add_argument("--data_version", type=str, default=DATA_VERSION)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    
    asyncio.run(main(args))

