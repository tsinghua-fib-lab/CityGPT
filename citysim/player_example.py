from pycityproto.city.geo.v2.geo_pb2 import (
    AoiPosition,
    LanePosition,
    Position,
    XYPosition,
    LongLatPosition,
)
import pycityproto.city.routing.v2.routing_pb2 as routing_pb
import pycityproto.city.routing.v2.routing_service_pb2 as routing_service
from pycitydata.map import Map
from .routing import RoutingClient

import math
from shapely.geometry import Point, Polygon
from typing import Tuple, cast,Optional, Union, Dict, Any
from .protobuf import async_parse, parse


def calculate_direction_and_distance(start_x, start_y, end_x, end_y):
    # 计算距离
    distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

    # 计算方位角
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    if delta_x == 0 and delta_y == 0:
        # 如果起点和终点相同，则方位无法确定
        direction = "同一位置"
    elif delta_x == 0:
        # 如果只在y轴上移动，则方位为北或南
        direction = "北" if delta_y > 0 else "南"
    else:
        # 计算方位角的弧度
        angle = math.atan(abs(delta_y / delta_x))
        # 将弧度转换为度
        angle_deg = math.degrees(angle)
        # 根据象限确定方位
        if delta_x > 0 and delta_y > 0:  # 第一象限
            direction = "东北" if angle_deg < 45 else "北"
        elif delta_x < 0 and delta_y > 0:  # 第二象限
            direction = "西北" if angle_deg < 45 else "北"
        elif delta_x < 0 and delta_y < 0:  # 第三象限
            direction = "西北" if angle_deg > 45 else "西"
        elif delta_x > 0 and delta_y < 0:  # 第四象限
            direction = "东北" if angle_deg > 45 else "东"

    return direction, distance


def filter_road_info(roads):
    # 初始化一个新的字典来存储结果
    filtered_roads = {}

    # 遍历输入的字典
    for road_id, info in roads.items():
        road_name, direction, distance = info
        # 创建一个复合键由road_name和direction组成
        key = (road_name, direction)

        # 如果这个复合键还没有在结果字典中，或者找到了更小的distance，则更新结果字典
        if key not in filtered_roads or distance < filtered_roads[key][2]:
            filtered_roads[key] = [road_name, direction, distance, road_id]

    # 由于我们存储了额外的road_id，我们需要重新整理结果字典，只保留road_id
    result = {info[-1]: info[:-1] for info in filtered_roads.values()}

    return dict(result)


class Player:
    def __init__(
        self,
        city_map: Map,
        city_routing_client: RoutingClient,
        init_aoi_id: int,
        search_type: str
    ):
        self._city_map = city_map
        self.search_type = search_type
        self._city_routing_client = city_routing_client

        self.init_position(init_aoi_id)
        self.time_cost = 0 
        self.price_cost = 0 

        self.current_road_list = []

    def init_position(self, init_aoi_id):
        aoi = self._city_map.get_aoi(init_aoi_id)
        if aoi is None:
            raise ValueError(f"aoi {init_aoi_id} not found")
        xy = cast(Polygon, aoi["shapely_xy"]).centroid
        x, y = xy.coords[0][0], xy.coords[0][1]
        lng, lat = self._city_map.xy2lnglat(x, y)
        if self._city_map.get_aoi(init_aoi_id)["driving_positions"]:
            lane_pos = self._city_map.get_aoi(init_aoi_id)["driving_positions"][0]
        else:
            lane_pos={"lane_id":0,"s":0}

        self.position = Position(
            aoi_position=AoiPosition(aoi_id=init_aoi_id),
            xy_position=XYPosition(x=x, y=y),
            longlat_position=LongLatPosition(longitude=lng, latitude=lat),
            lane_position=LanePosition(lane_id=lane_pos["lane_id"], s=lane_pos["s"])
        )
    

    def get_position(self):
        """
        获取当前位置
        """
        return parse(self.position, True)
    
    def lnglat2xy(self, lng: float, lat: float) -> Tuple[float, float]:
        """
        经纬度转xy坐标
        Convert latitude and longitude to xy coordinates

        Args:
        - lng (float): 经度。longitude.
        - lat (float): 纬度。latitude.

        Returns:
        - Tuple[float, float]: xy坐标。xy coordinates.
        """
        return self.projector(lng, lat)

    def search(
        self,
        center:  Union[Tuple[float, float], Point],
        radius: float,
        category_prefix: str,
        limit: int = 10,
    ):
        """
        搜索给定范围内的POI
        """
        if self.search_type=="poi":
            return self._city_map.query_pois(center, radius, category_prefix, limit)
        elif self.search_type=="aoi":
            return self._city_map.query_aois(center, radius, category_prefix, limit)

    async def get_walking_route(self, aoi_id: int):
        """
        获取步行路线和代价
        """
        print(f"get_walking_route: {self.position.aoi_position.aoi_id} -> {aoi_id}")
        resp = await self._city_routing_client.GetRoute(
            routing_service.GetRouteRequest(
                type=routing_pb.ROUTE_TYPE_WALKING,
                start=self.position,
                end=Position(aoi_position=AoiPosition(aoi_id=aoi_id)),
            ),
            dict_return=False,
        )
        resp = cast(routing_service.GetRouteResponse, resp)
        if len(resp.journeys) == 0:
            return None
        return parse(resp.journeys[0].walking, True)

    async def get_driving_route(self, aoi_id: int):
        """
        获取开车路线和代价
        """
        # print(f"get_driving_route: {self.position.aoi_position.aoi_id} -> {aoi_id}")
        resp = await self._city_routing_client.GetRoute(
            routing_service.GetRouteRequest(
                type=routing_pb.ROUTE_TYPE_DRIVING,
                start=self.position,
                end=Position(aoi_position=AoiPosition(aoi_id=aoi_id)),
            ),
            dict_return=False,
        )
        resp = cast(routing_service.GetRouteResponse, resp)
        if len(resp.journeys) == 0:
            return None
        return parse(resp.journeys[0].driving, True)

    async def walk_to(self, aoi_id: int) -> bool:
        """
        步行到POI，实际产生移动，更新人物位置
        """
        # 起终点相同，直接返回True
        if aoi_id == self.position.aoi_position.aoi_id:
            return True
        
        route = await self.get_walking_route(aoi_id)
        if route is None:
            return False
        last_lane_id = route["route"][-1]["lane_id"]
        # 检查对应的AOI Gate
        aoi = self._city_map.get_aoi(aoi_id)
        if aoi is None:
            raise ValueError(f"aoi {aoi_id} not found")
        gate_index = -1
        for i, p in enumerate(aoi["walking_positions"]):
            if p["lane_id"] == last_lane_id:
                gate_index = i
                break
        if gate_index == -1:
            raise ValueError(
                f"aoi {aoi_id} has no walking gate for lane {last_lane_id}"
            )
        # 更新人物位置
        gate_xy = aoi["walking_gates"][gate_index]
        x, y = gate_xy["x"], gate_xy["y"]
        lng, lat = self._city_map.xy2lnglat(x, y)
        self.position = Position(
            aoi_position=AoiPosition(aoi_id=aoi_id),
            xy_position=XYPosition(x=x, y=y),
            longlat_position=LongLatPosition(longitude=lng, latitude=lat),
        )
        self.time_cost += route["eta"]
        self.price_cost += 0
        return True

    async def drive_to(self, aoi_id: int):
        """
        开车到POI，实际产生移动，更新人物位置
        """
         # 起终点相同，直接返回True
        if aoi_id == self.position.aoi_position.aoi_id:
            return True
        
        route = await self.get_driving_route(aoi_id)
        if route is None:
            return False
        last_road_id = route["road_ids"][-1]
        # 检查对应的AOI Gate
        aoi = self._city_map.get_aoi(aoi_id)
        if aoi is None:
            raise ValueError(f"aoi {aoi_id} not found")
        gate_index = -1
        for i, p in enumerate(aoi["driving_positions"]):
            lane_id = p["lane_id"]
            lane = self._city_map.get_lane(lane_id)
            if lane is None:
                raise ValueError(f"lane {lane_id} not found")
            road_id = lane["parent_id"]
            if road_id == last_road_id:
                gate_index = i
                break
        if gate_index == -1:
            raise ValueError(
                f"aoi {aoi_id} has no driving gate for road {last_road_id}"
            )
        # 更新人物位置
        gate_xy = aoi["driving_gates"][gate_index]
        x, y = gate_xy["x"], gate_xy["y"]
        lng, lat = self._city_map.xy2lnglat(x, y)
        self.position = Position(
            aoi_position=AoiPosition(aoi_id=aoi_id),
            xy_position=XYPosition(x=x, y=y),
            longlat_position=LongLatPosition(longitude=lng, latitude=lat),
        )
        self.time_cost += route["eta"]
        self.price_cost += route["eta"]
        return True

    def get_aoi_of_poi(self, poi_id):
        if poi_id in self._city_map.pois:
            aoi_id = self._city_map.get_poi(poi_id)["aoi_id"]
            return aoi_id
        else:
            print("POI:{}的没有AOI归属信息信息".format(poi_id))
            return None
    
    async def set_routing_list(self, poi_id):
        aoi_id = self.get_aoi_of_poi(poi_id)
        self.current_road_list = await self.get_driving_route(aoi_id)
    
    async def move_step_by_step(self):
        """执行单步移动操作"""
        
        # 获取当前道路
        current_lane_id = self.position.lane_position.lane_id
        parent_id = self._city_map.get_lane(current_lane_id)["parent_id"]
        try:
            if "external" in self._city_map.get_road(parent_id):
                name = self._city_map.get_road(parent_id)["external"]["name"]
            else:
                name = self._city_map.get_road(parent_id)["name"]
            print("current_road:",name)
        except:
            print("notavailable")

        try:
            pre_lane_id = self._city_map.get_lane(current_lane_id)["predecessors"][0]["id"]
        except IndexError as e:
            return (False, "No avaiable lanes")
        
        pre_lane_info = self._city_map.get_lane(pre_lane_id)

        # 获取路口
        junc_id = pre_lane_info["parent_id"]
        junc_info = self._city_map.get_junction(junc_id)

        # 获取可行路口
        avaiable_lanes = []
        for junc_lane_id in junc_info["lane_ids"]:
            junc_lane_info = self._city_map.get_lane(junc_lane_id)

            for predecessor in junc_lane_info["predecessors"]:
                junc_pre_lane_id = predecessor["id"]
                avaiable_lanes.append(junc_pre_lane_id)
        avaiable_road_names = {}
        for lane_id in avaiable_lanes:
            parent_id = self._city_map.get_lane(lane_id)["parent_id"]
            try:
                #name = self._city_map.get_road(parent_id)["external"]["name"]
                name = self._city_map.get_road(parent_id)["name"]
            except:
                name = "unknown"
            avaiable_road_names[lane_id] = name
        
        # 选择一条路前行
        next_lane_id = avaiable_lanes[0]

        lane_info = self._city_map.get_lane(next_lane_id)
        endpoint_lnglat = lane_info["shapely_lnglat"].coords[-1]
        endpoint_xy = lane_info["shapely_xy"].coords[-1]

        # 发生移动，更新位置
        self.position = Position(
            xy_position=XYPosition(x=endpoint_xy[0], y=endpoint_xy[1]),
            longlat_position=LongLatPosition(longitude=endpoint_lnglat[0], latitude=endpoint_lnglat[1]),
            lane_position=LanePosition(lane_id=next_lane_id, s=0)
        )

        return (True, "Move One Step")
    
    def get_junction_list(self):
        # 获取当前道路与当前位置
        current_lane_id = self.position.lane_position.lane_id
        current_xy=dict(self.get_position()["xy_position"])
        current_lnglat=dict(self.get_position()["longlat_position"])
        #print("cp",current_lnglat)
        #print("current_lane_id",current_lane_id)
        try:
            pre_lane_id = self._city_map.get_lane(current_lane_id)["predecessors"][0]["id"]
        except IndexError as e:
            return (False, "No avaiable lanes")
        
        pre_lane_info = self._city_map.get_lane(pre_lane_id)

        # 获取路口
        junc_id = pre_lane_info["parent_id"]
        junc_info = self._city_map.get_junction(junc_id)

        # 获取可行路口
        avaiable_lanes = []
        for junc_lane_id in junc_info["lane_ids"]:
            junc_lane_info = self._city_map.get_lane(junc_lane_id)
            for predecessor in junc_lane_info["predecessors"]:
                junc_pre_lane_id = predecessor["id"]
                #此处添加对lane方向与距离的计算
                lane_info = self._city_map.get_lane(junc_pre_lane_id)
                #1.得到行走一步以后的位置
                endpoint_lnglat_temp = lane_info["shapely_lnglat"].coords[-1]
                endpoint_lnglat={"longitude":endpoint_lnglat_temp[0], "latitude":endpoint_lnglat_temp[1]}
                endpoint_xy_temp = lane_info["shapely_xy"].coords[-1]
                endpoint_xy={'x':endpoint_xy_temp[0],'y':endpoint_xy_temp[1]}
                #计算lane对应的方向与距离
                dir,dis=calculate_direction_and_distance(current_xy['x'], current_xy['y'], endpoint_xy['x'],endpoint_xy['y'])

                avaiable_lanes.append([junc_pre_lane_id,dir,dis])
        available_road_names = {}
        for lane in avaiable_lanes:
            parent_id = self._city_map.get_lane(lane[0])["parent_id"]
            try:
                #name = self._city_map.get_road(parent_id)["external"]["name"]
                name = self._city_map.get_road(parent_id)["name"]
            except:
                name = "unknown"
            available_road_names[lane[0]] = [name,lane[1],lane[2]]
        #只保留距离最短的lane
        filtered_info=filter_road_info(available_road_names)
        return filtered_info

    #player根据选择前进
    def move_after_decision(self,next_lane_id):
        lane_info = self._city_map.get_lane(next_lane_id)
        endpoint_lnglat = lane_info["shapely_lnglat"].coords[-1]
        endpoint_xy = lane_info["shapely_xy"].coords[-1]

        # 发生移动，更新位置
        self.position = Position(
            xy_position=XYPosition(x=endpoint_xy[0], y=endpoint_xy[1]),
            longlat_position=LongLatPosition(longitude=endpoint_lnglat[0], latitude=endpoint_lnglat[1]),
            lane_position=LanePosition(lane_id=next_lane_id, s=0)
        )

    def check_position(self,end_xy,thres):
        current_xy=dict(self.get_position()["xy_position"])
        start_x=current_xy['x']
        start_y=current_xy['y']
        end_x=end_xy['x']
        end_y=end_xy['y']
        distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        #print("updated distance",distance)
        """确认是否已到达指定位置"""
        if distance<thres:
            return 0
        else:
            return distance


    def get_cur_position(self):
        current_xy=dict(self.get_position()["xy_position"])
        start_x=current_xy['x']
        start_y=current_xy['y']
        if self.search_type=="aoi":
            list_temp=self.search([start_x,start_y],1000,["E3"],2)
        elif self.search_type=="poi":
            list_temp=self.search([start_x,start_y],1000,"",2)
        poi_list=[]
        for items in list_temp:
            poi_list.append(items[0]['name'])
        return poi_list
