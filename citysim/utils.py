import re
import os
import logging
import random
import pyproj
import asyncio
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import Optional, cast, List, Tuple, Dict
from multiprocessing import Pool,cpu_count

from mosstool.trip.generator.generate_from_od import TripGenerator
from mosstool.trip.route import RoutingClient
from mosstool.type import Map, Person, Persons
from mosstool.util.format_converter import dict2pb,pb2dict

from pycityproto.city.person.v1.person_pb2 import Person
from pycityproto.city.routing.v2.routing_pb2 import RouteType
from pycityproto.city.routing.v2.routing_service_pb2 import GetRouteRequest
from pycityproto.city.trip.v2.trip_pb2 import Schedule, TripMode
LANE_TYPE_DRIVE = 1
LANE_TYPE_WALK = 2
_TYPE_MAP = {
    TripMode.TRIP_MODE_DRIVE_ONLY: RouteType.ROUTE_TYPE_DRIVING,
    TripMode.TRIP_MODE_BIKE_WALK: RouteType.ROUTE_TYPE_WALKING,
    TripMode.TRIP_MODE_BUS_WALK: RouteType.ROUTE_TYPE_WALKING,
    TripMode.TRIP_MODE_WALK_ONLY: RouteType.ROUTE_TYPE_WALKING,
}

from config import MAP_DATA_PATH, SIGNAL_BOX


def generate_persons(m: Map, city: str, agent_num: Optional[int] = None):
    city_data_path = os.path.join(MAP_DATA_PATH, city)
    area = gpd.read_file(os.path.join(city_data_path, f"{city}.shp"))
    od_path = os.path.join(city_data_path, f"{city}_od.npy")
    od_matrix = np.load(od_path)
    tg = TripGenerator(
        m=m,
    )
    if agent_num is None:
        od_persons = tg.generate_persons(
            od_matrix=od_matrix,
            areas=area,
            seed=0,
        )
    else:
        od_persons = tg.generate_persons(
            od_matrix=od_matrix,
            areas=area,
            seed=0,
            agent_num=agent_num,
        )
    return od_persons


# ATTENTION:和pre_route不同在于找到一个非空schedule就会返回
async def my_pre_route(
    client: RoutingClient, person: Person, in_place: bool = False
) -> Person:
    if not in_place:
        p = Person()
        p.CopyFrom(person)
        person = p
    start = person.home
    departure_time = None
    all_schedules = list(person.schedules)
    person.ClearField("schedules")
    for schedule in all_schedules:
        schedule = cast(Schedule, schedule)
        if schedule.HasField("departure_time"):
            departure_time = schedule.departure_time
        if schedule.loop_count != 1:
            logging.warning(
                "Schedule is not a one-time trip, departure time is not accurate, no pre-calculation is performed"
            )
            start = schedule.trips[-1].end
            continue
        good_trips = []
        for trip in schedule.trips:
            last_departure_time = departure_time
            # Cover departure time
            if trip.HasField("departure_time"):
                departure_time = trip.departure_time
            if departure_time is None:
                continue
            # ATTENTION:只保留行车
            if not trip.mode == TripMode.TRIP_MODE_DRIVE_ONLY:
                departure_time = last_departure_time
                continue
            # build request
            res = await client.GetRoute(
                GetRouteRequest(
                    type=_TYPE_MAP[trip.mode],
                    start=start,
                    end=trip.end,
                    time=departure_time,
                )
            )
            if len(res.journeys) == 0:
                # logging.warning("No route found")
                departure_time = last_departure_time
            else:
                # append directly
                good_trips.append(trip)
                trip.ClearField("routes")
                trip.routes.MergeFrom(res.journeys)
                # update start position
                start = trip.end
                # Set departure time invalid
                departure_time = None
        if len(good_trips) > 0:
            good_trips = good_trips[:1]
            good_schedule = cast(Schedule, person.schedules.add())
            good_schedule.CopyFrom(schedule)
            good_schedule.ClearField("trips")
            good_schedule.trips.extend(good_trips)
            break
    return person

def pos_transfer_unit(dict_p):
    global m_lanes,m_roads,m_aois
    len_schedules = len(dict_p["schedules"])
    if len_schedules>2:
        select_idx = random.choice([i for i in range(1,len_schedules-1)])
        pre_end = dict_p["schedules"][select_idx-1]["trips"][0]["end"]
        dict_p["schedules"] = dict_p["schedules"][select_idx:]
        dict_p["home"] = pre_end
    ##
    IS_BAD_PERSON = False
    aoi_id = dict_p["home"]["aoi_position"]["aoi_id"]
    aoi = m_aois[aoi_id]
    trip_mode = dict_p["schedules"][0]["trips"][0]["mode"]
    all_aoi_lane_ids = list(
        [pos["lane_id"] for pos in aoi["walking_positions"]]
        + [pos["lane_id"] for pos in aoi["driving_positions"]]
    )
    extra_lane_ids = []
    for lid in all_aoi_lane_ids:
        for pre in m_lanes[lid]["predecessors"]:
            extra_lane_ids.append(pre["id"])
        for suc in m_lanes[lid]["successors"]:
            extra_lane_ids.append(suc["id"])
    all_aoi_lane_ids += extra_lane_ids
    all_aoi_road_ids = [m_lanes[lid]["parent_id"] for lid in all_aoi_lane_ids]
    all_aoi_road_ids = list(
        set([pid for pid in all_aoi_road_ids if pid < 3_0000_0000])
    )
    all_aoi_drive_lane_ids = [lid for rid in all_aoi_road_ids for lid in m_roads[rid]["lane_ids"] if m_lanes[lid]["type"] == LANE_TYPE_DRIVE] 
    all_aoi_walk_lane_ids = [lid for rid in all_aoi_road_ids for lid in m_roads[rid]["lane_ids"] if m_lanes[lid]["type"] == LANE_TYPE_WALK] 
    select_lane_id = None
    if trip_mode in [
        TripMode.TRIP_MODE_BIKE_WALK,
        TripMode.TRIP_MODE_BUS_WALK,
        TripMode.TRIP_MODE_WALK_ONLY,
    ]:
        if len(all_aoi_walk_lane_ids)>0:
            select_lane_id = random.choice(all_aoi_walk_lane_ids)
        else:
            IS_BAD_PERSON = True
    else:
        if len(all_aoi_drive_lane_ids)>0:
            select_lane_id = random.choice(all_aoi_drive_lane_ids)
        else:
            IS_BAD_PERSON = True
    if IS_BAD_PERSON:
        return None
    select_lane = m_lanes[select_lane_id]
    dict_p["home"] = {
        "lane_position":{
            "lane_id":select_lane["id"],
            "s":random.uniform(0.1,0.9) * select_lane["length"],
        }
    }
    # 每个schedule只有一个trip
    for schedule in dict_p["schedules"]:
        for trip in schedule["trips"]:
            if IS_BAD_PERSON:
                continue
            aoi_id = trip["end"]["aoi_position"]["aoi_id"]
            aoi = m_aois[aoi_id]
            all_aoi_lane_ids = list(
                [pos["lane_id"] for pos in aoi["walking_positions"]]
                + [pos["lane_id"] for pos in aoi["driving_positions"]]
            )
            extra_lane_ids = []
            for lid in all_aoi_lane_ids:
                for pre in m_lanes[lid]["predecessors"]:
                    extra_lane_ids.append(pre["id"])
                for suc in m_lanes[lid]["successors"]:
                    extra_lane_ids.append(suc["id"])
            all_aoi_lane_ids += extra_lane_ids
            all_aoi_road_ids = [m_lanes[lid]["parent_id"] for lid in all_aoi_lane_ids]
            all_aoi_road_ids = list(
                set([pid for pid in all_aoi_road_ids if pid < 3_0000_0000])
            )
            all_aoi_drive_lane_ids = [lid for rid in all_aoi_road_ids for lid in m_roads[rid]["lane_ids"] if m_lanes[lid]["type"] == LANE_TYPE_DRIVE] 
            all_aoi_walk_lane_ids = [lid for rid in all_aoi_road_ids for lid in m_roads[rid]["lane_ids"] if m_lanes[lid]["type"] == LANE_TYPE_WALK] 
            select_lane_id = None
            if trip_mode in [
                TripMode.TRIP_MODE_BIKE_WALK,
                TripMode.TRIP_MODE_BUS_WALK,
                TripMode.TRIP_MODE_WALK_ONLY,
            ]:
                if len(all_aoi_walk_lane_ids)>0:
                    select_lane_id = random.choice(all_aoi_walk_lane_ids)
                else:
                    IS_BAD_PERSON = True
            else:
                if len(all_aoi_drive_lane_ids)>0:
                    select_lane_id = random.choice(all_aoi_drive_lane_ids)
                else:
                    IS_BAD_PERSON = True
            if IS_BAD_PERSON:
                continue
            select_lane = m_lanes[select_lane_id]
            trip["end"] = {
                "lane_position":{
                    "lane_id":select_lane["id"],
                    "s":random.uniform(0.1,0.9) * select_lane["length"],
                }
            }
    if IS_BAD_PERSON:
        return None
    else:
        return dict_p

def multi_aoi_pos2lane_pos(persons, map_dict):
    global m_lanes,m_roads,m_aois
    workers = cpu_count()
    persons_pb = Persons(persons=persons)
    persons_list = pb2dict(persons_pb)["persons"]
    m_aois = {d["id"]: d for d in map_dict["aois"]}
    m_lanes = {d["id"]: d for d in map_dict["lanes"]}
    m_roads = {d["id"]: d for d in map_dict["roads"]}
    new_persons = []
    MAX_BATCH_SIZE = 15_0000
    for i in tqdm(range(0, len(persons_list), MAX_BATCH_SIZE)):
        persons_batch = persons_list[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            new_persons += pool.map(
                pos_transfer_unit,
                persons_batch,
                chunksize=min(len(persons_batch) // workers, 1000),
            )
    new_persons = [dict2pb(p,Person()) for p in new_persons if p is not None]
    return new_persons
async def with_preroute(persons, lanes, listen: str, output_path: str,max_num:int):
    client = RoutingClient(listen)
    all_persons = []
    BATCH = 15000
    for i in tqdm(range(0, len(persons), BATCH)):
        ps = await asyncio.gather(
            *[my_pre_route(client, p) for p in persons[i : i + BATCH]]
        )
        all_persons.extend(ps)

    ok_persons = []
    for p in all_persons:
        if len(p.schedules) == 0:
            continue
        if len(p.schedules[0].trips) == 0:
            continue
        BAD_PERSON = False
        start_id = p.home.lane_position.lane_id
        end_id = p.schedules[0].trips[0].end.lane_position.lane_id
        trip_mode = p.schedules[0].trips[0].mode
        if trip_mode in [
            TripMode.TRIP_MODE_BIKE_WALK,
            TripMode.TRIP_MODE_BUS_WALK,
            TripMode.TRIP_MODE_WALK_ONLY,
        ]:
            if (
                not lanes[start_id]["type"] == LANE_TYPE_WALK
                or not lanes[end_id]["type"] == LANE_TYPE_WALK
            ):
                BAD_PERSON = True
        else:
            if (
                not lanes[start_id]["type"] == LANE_TYPE_DRIVE
                or not lanes[end_id]["type"] == LANE_TYPE_DRIVE
            ):
                BAD_PERSON = True
        if BAD_PERSON:
            continue
        ok_persons.append(p)
    ok_persons = ok_persons[:max_num]
    for ii, p in enumerate(ok_persons):
        p.id = ii
    pb = Persons(persons=ok_persons)
    with open(output_path, "wb") as f:
        f.write(pb.SerializeToString())

def get_coords(city: str) -> List[Tuple[float, float]]:
    # 根据城市名返回坐标列表。
    coords_dict: Dict[str, List[Tuple[float, float]]] = SIGNAL_BOX
    return coords_dict.get(city, [])


# 对于多个junc，记录phase状态
class JunctionState:
    def __init__(self):
        self.phase_index = 0


# 解析response的相位信息
def validate_response(response, phase_map):
    # Extracting the number enclosed in <signal> tags
    match = re.search(r'<signal>.*?(\d+).*?</signal>', response)
    if match:
        chosen_number = int(match.group(1))
        # Check if the chosen number is within the valid range
        if chosen_number in phase_map:
            return True, chosen_number
        else:
            return False, "Invalid phase option number."
    return False, "No valid phase option number provided."



def process_phase(j, green_indices, filtered_index, cnt, waiting_cnt):
    phase_info = {}
    green_pre_lanes = set()
    vehicle_counts = 0
    waiting_vehicle_counts = 0
    for green_index in green_indices:
        pre_lane_index = j.lanes[green_index].predecessors[0].index
        green_pre_lanes.add(pre_lane_index)

    for green_pre_lane in green_pre_lanes:
        
        vehicle_counts += cnt[green_pre_lane]
        waiting_vehicle_counts += waiting_cnt[green_pre_lane]

    lane_counts = len(green_pre_lanes)
    return (filtered_index, lane_counts, vehicle_counts, waiting_vehicle_counts)


def get_chosen_number_from_phase_index(phase_index, phase_map):
    for chosen_number, current_phase_index in phase_map.items():
        if current_phase_index == phase_index:
            return chosen_number
    print(f"phase_index: {phase_index}")
    print(f"phase_map: {phase_map}")
    return None  # 如果没有找到相应的chosen_number，返回None


def lnglat2xy(M, coords):
    proj_str = M.pb.header.projection
    projector = pyproj.Proj(proj_str)
    coords_array = np.array(coords)
    xs, ys = projector(coords_array[:, 0], coords_array[:, 1])
    xy_coords = list(zip(xs, ys))
    return xy_coords

def whether_road_in_region(M, coords):
    # 判断road是否在划分的区域内
    # 将经纬度转换为平面坐标并存储到列表
    xy_coords = lnglat2xy(M, coords)
    # print(xy_coords)
    target_road_ids = []
    # 获取区域内道路
    polygon = Polygon(xy_coords)
    for road in M.roads:
        # 根据road的第一条lane判断road是否在区域内
        first_lane_id = road.pb.lane_ids[0]
        first_lane = M.lane_map[first_lane_id]
        x_coords = [node.x for node in first_lane.pb.center_line.nodes]
        y_coords = [node.y for node in first_lane.pb.center_line.nodes]

        # 计算 x 和 y 坐标的平均值
        x_mean = sum(x_coords) / len(x_coords)
        y_mean = sum(y_coords) / len(y_coords)
        point = Point(x_mean, y_mean)
        if polygon.contains(point):
            target_road_ids.append(road.pb.id)
            print(road.pb.name)
    return target_road_ids


def whether_junc_in_region(M, j, coords):
    # 判断junction是否在划分的区域内
    xy_coords = lnglat2xy(M, coords)
    polygon = Polygon(xy_coords)
    
    first_lane_id = j.pb.lane_ids[0]
    first_lane = M.lane_map[first_lane_id]
    x_coords = [node.x for node in first_lane.pb.center_line.nodes]
    y_coords = [node.y for node in first_lane.pb.center_line.nodes]

    # 计算 x 和 y 坐标的平均值
    x_mean = sum(x_coords) / len(x_coords)
    y_mean = sum(y_coords) / len(y_coords)
    point = Point(x_mean, y_mean)
    if polygon.contains(point):
        return True
    else:
        return False
