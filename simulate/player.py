import random 
import copy
import os
import sys
import json
from typing import Tuple, cast
from shapely.geometry import Point, Polygon

# 城市模拟器相关代码
from pycityproto.city.geo.v2.geo_pb2 import AoiPosition, Position, XYPosition, LongLatPosition
import pycityproto.city.routing.v2.routing_pb2 as routing_pb
import pycityproto.city.routing.v2.routing_service_pb2 as routing_service
from pycitydata.map import Map
from citysim.routing import RoutingClient

from config import EVAL_DATA, LANDMARK_DATA, VISION_DATA, REGION_BOUNDARY
from simulate.utils import *
from simulate.templates import current_aoi_position, start_dest_text, junc_name_text, junc_walk_text, describe_pois_via_text,  describe_one_step_text, end_point_text, final_position, step_interests_text
from simulate.translate import Name



#####################
# 下面一行失败，因此直接将相关代码copy过来
# from citysim.utils.protobuf import parse
from typing import Any, Awaitable, TypeVar
from google.protobuf.message import Message
from google.protobuf.json_format import MessageToDict

T = TypeVar("T", bound=Message)

def parse(res: T, dict_return: bool) -> dict[str, Any] | T:
    """
    将Protobuf返回值转换为dict或者原始值
    """
    if dict_return:
        return MessageToDict(
            res,
            including_default_value_fields=True,
            preserving_proto_field_name=True,
            use_integers_for_enums=True,
        )
    else:
        return res


async def async_parse(res: Awaitable[T], dict_return: bool) -> dict[str, Any] | T:
    """
    将Protobuf await返回值转换为dict或者原始值
    """
    if dict_return:
        return MessageToDict(
            await res,
            including_default_value_fields=True,
            preserving_proto_field_name=True,
            use_integers_for_enums=True,
        )
    else:
        return await res

def category_mapping():
    # 定义优先关注的POI类别
    category_supported = {"leisure":"leisure", "amenity":"amenity", "building":"building"}

    return category_supported

######################

class Player:
    def __init__(
        self,
        city_map: Map,
        city_routing_client: RoutingClient,
        init_aoi_id: int,
    ):
        self._city_map = city_map
        self._city_routing_client = city_routing_client

        self.init_position(init_aoi_id)
        # TODO 似乎缺失一个POI position 判断，是否已经在postion里面
        self.time_cost = 0  # 总时间代价
        self.price_cost = 0 # 总成本代价

    def init_position(self, init_aoi_id):
        aoi = self._city_map.get_aoi(init_aoi_id)
        if aoi is None:
            raise ValueError(f"aoi {init_aoi_id} not found")
        xy = cast(Polygon, aoi["shapely_xy"]).centroid
        x, y = xy.coords[0][0], xy.coords[0][1]
        lng, lat = self._city_map.xy2lnglat(x, y)

        self.position = Position(
            aoi_position=AoiPosition(aoi_id=init_aoi_id),
            xy_position=XYPosition(x=x, y=y),
            longlat_position=LongLatPosition(longitude=lng, latitude=lat),
        )  # 当前位置
    
    # 可执行的动作列表

    def get_position(self):
        """
        获取当前位置
        """
        return parse(self.position, True)

    def get_time_cost(self):
        """
        获取当前时间代价
        """
        return self.time_cost

    def get_price_cost(self):
        return self.price_cost
    
    def search(
        self,
        center: Tuple[float, float] | Point,
        radius: float,
        category_prefix: str,
        limit: int = 10,
    ):
        """
        搜索给定范围内的POI
        """
        return self._city_map.query_pois(center, radius, category_prefix, limit)

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

    def look_indoor(self, limit=10):
        """
        在AOI内当前位置观察，返回最近的POI列表
        """
        aoi_id = self.position.aoi_position.aoi_id
        aoi = self._city_map.get_aoi(aoi_id)
        if aoi is None:
            raise ValueError(f"aoi {aoi_id} not found")
        poi_ids = aoi["poi_ids"]
        # 获取AOI内所有POI
        pois = {}
        for poi_id in poi_ids:
            poi = self._city_map.get_poi(poi_id)
            if poi is None:
                raise ValueError(f"poi {poi_id} not found")
            pois[poi_id] = poi
        # 计算当前位置到每个poi的位置
        center = Point(self.position.xy_position.x, self.position.xy_position.y)
        poi_distance = []
        for poi in pois.values():
            distance = center.distance(poi["shapely_xy"])
            poi_distance.append((poi, distance))
        # 按距离排序
        poi_distance.sort(key=lambda x: x[1])
        # 返回最近的limit个POI
        return [poi for poi, _ in poi_distance[:limit]]

    def move_indoor(self, poi_id: int):
        """
        在AOI内移动，更新人物位置
        """
        poi = self._city_map.get_poi(poi_id)
        if poi is None:
            raise ValueError(f"poi {poi_id} not found")
        if poi["aoi_id"] != self.position.aoi_position.aoi_id:
            raise ValueError(f"poi {poi_id} not in aoi {self.position.aoi_position.aoi_id}")
        x, y = poi["position"]["x"], poi["position"]["y"]
        lng, lat = self._city_map.xy2lnglat(x, y)
        self.position = Position(
            aoi_position=AoiPosition(aoi_id=self.position.aoi_position.aoi_id, poi_id=poi_id),
            xy_position=poi["position"],
            longlat_position=LongLatPosition(longitude=lng, latitude=lat),
        )
        return True

    def do(self, function: str):
        """
        使用当前所在的POI的服务，返回成功与否
        """
        # TODO: 根据poi的种类判定其所能支持的服务，检查与传入的是否一致
        ...


class TextPlayer(Player):
    def __init__(
        self,
        city_map: Map,
        city_routing_client: RoutingClient,
        init_aoi_id: int,
        min_road_length: int,
        region_exp: str,
        nearby_params={"radius": 100, "limit": 10, "has_category": 0},
        init_poi_id = None
    ):
        """
        类比virtual-home: region->home, aoi->room, poi->object, service->action
        """
        self.init_aoi_id = init_aoi_id
        if init_poi_id is not None:
            self.init_poi_id = init_poi_id
        self.min_road_length = min_road_length
        super().__init__(
            city_map=city_map,
            city_routing_client=city_routing_client,
            init_aoi_id=init_aoi_id
        )
        self.LANGUAGE = Name(region_exp=region_exp)
        
        # 定义实验区域region，限定可用POI，AOI和路网，small和large对应同一个区域的不同范围，统一修改为small
        self.region_exp_dict = REGION_BOUNDARY
        self.region_exp_polygon = Polygon(self.region_exp_dict[region_exp])
        self.region_center = self.region_exp_polygon.centroid.x, self.region_exp_polygon.centroid.y
        ########################################

        ########################################
        # 下面为不断更新的环境状态，在reset阶段需要刷新 

        # 至今所有步骤中所看到的POI信息，对于POI在实验区域内进行局部编号，类比home中有多个物品
        self.visiable_pois, self.visiable_pois_revserse = {}, {}
        self.visiable_pois_global = {}

        # 维护完成任务的代价，此外还包括time_cost, price_cost
        self.totlal_steps = 0

        # 任务强行结束的条件，运行超过30步
        self.max_episode_length = 10

        # 当前步骤所见看到的POI信息
        self.current_step_known_pois_local_view = []    # update after search
        self.current_step_visiable_pois_local_view = [] # updare after look(indoor), utilized by move(indoor)

        self.script_object = None
        self.script_action = None
        self.current_action_space = None

        # CityWalk闲逛时，产生的路径信息
        self.current_road_list = []
        self.routing_road_list = []
        self.routing_junction_list = []

        self.register_poi = {}

        self.nearby_params = nearby_params

    
    def get_step_cost(self):
        return self.totlal_steps


    def check_in_region(self, shapely_point: Point):
        # 确认关键元素POI/AOI/Road是否在实验区域内
        return self.region_exp_polygon.contains(shapely_point)


    def set_max_episode_length(self, number: int):
        self.max_episode_length = number

    def get_max_episode_length(self):
        return self.max_episode_length

   
    ############ 维护到目前为止，所有被发现过的POI信息，并给予local id
    # 区分通过线上搜索或者局部视角来确认当前可访问POI列表
    def update_current_step_visiable_pois(self, poi_list):
        if self.script_action == Action.LOOK_IN_DOOR.value:
            self.current_step_visiable_pois_local_view = []
            for poi in poi_list:
                self.current_step_visiable_pois_local_view.append(self.from_global_to_local(poi))
        elif self.script_action == Action.SEARCH.value:
            self.current_step_known_pois_local_view = []
            for poi in poi_list:
                self.current_step_known_pois_local_view.append(self.from_global_to_local(poi))
        elif len(poi_list)==0:
            self.current_step_known_pois_local_view = []
        else:
            raise NotImplementedError


    # 获取当前动作可见的pois列表，主要区分室内和室外，以诱导模型使用正确的功能
    def get_current_step_visiable_pois(self, action=Action.NAVIGATE.value):
        # TODO need to update, random walk应该考虑前序动作的影响
        if action in [Action.MOVE_IN_DOOR.value, Action.LOOK_IN_DOOR.value, Action.EXPLORE.value]:
            return self.current_step_visiable_pois_local_view
        else:
            return self.current_step_known_pois_local_view


    # 将当前可见的POI记录下来，并进行局部编码，方便随时基于局部编码反查全局ID进行模拟器操作
    def update_pois_in_visiable_dict(self, poi_list):
        # 记录新增POI，赋予category下的局部编码local_id
        for poi in poi_list:
            poi_id = poi["id"]
            category_id = poi["category"]
            if category_id not in self.visiable_pois:
                self.visiable_pois[category_id] = {poi_id: str(0)}
            else:
                if poi_id not in self.visiable_pois[category_id]:
                    local_id = str(len(self.visiable_pois[category_id])+1)
                    self.visiable_pois[category_id][poi_id] = local_id
        
        # 更新category_id+local_id和poi_id的映射字典
        for category_id in self.visiable_pois:
            if category_id not in self.visiable_pois_revserse:
                self.visiable_pois_revserse[category_id] = {}
            for poi_id in self.visiable_pois[category_id]:
                local_id = self.visiable_pois[category_id][poi_id]
                if local_id not in self.visiable_pois_revserse:
                    self.visiable_pois_revserse[category_id][local_id] = poi_id
        
        for poi in poi_list:
            poi_id = poi["id"]
            aoi_id = poi["aoi_id"]
            if poi_id not in self.visiable_pois_global:
                self.visiable_pois_global[poi_id] = self.from_global_to_local(poi)


    # 获取当前已知的所有POI信息，以局部ID表示
    def get_all_visiable_pois_local_view(self):
        return list(self.visiable_pois_global.values())
    # 获取当前已知的所有POI信息，以全局ID表示
    def get_all_visiable_pois_global_view(self):
        return list(self.visiable_pois_global.items())


    # 查询POI所属的AOI
    def get_aoi_of_poi(self, poi_id):
        if poi_id in self._city_map.pois:
            aoi_id = self._city_map.get_poi(poi_id)["aoi_id"]
            return aoi_id
        else:
            print("POI:{}的没有AOI归属信息信息".format(poi_id))
            return None

    
    def from_global_to_local(self, poi):
        category_id3 = poi["category"]
        category_name = self.get_text_of_category(category_id3)
        local_id = self.visiable_pois[category_id3][poi["id"]]
        text_local_id = category_name+'-'+str(local_id)
        return text_local_id

    
    def from_local_to_global(self, text_local_id):
        try:
            category_text, poi_local_id = text_local_id.split("-")
            category_id3 = self.get_category_id_from_text(category_text)
            poi_id = self.visiable_pois_revserse[category_id3][poi_local_id]

            return poi_id
        except Exception as e:
            print(e)
            raise NotImplementedError


    ##########################
    # Action1: Search
    # TODO 需要增加对POI类别的过滤，防止召回都是无用的POI
    # 不同于virtual home，在city中通过定向搜索来建立环境认知
    # 本质上，search是对手机端搜索功能的近似，但实现起来过于复杂，因此对其进行简化，限定为基于有限三级类目下的search
    def search(self, category: str, shuffle=False, clean=False):
        # 自我定位
        pos = self.get_position()
        center = (pos["xy_position"]["x"], pos["xy_position"]["y"])

        # TODO 自然语言描述的类别转换为内部代码前缀，这里假设只搜1级类目，后续可以拓展
        temp_cut = True # 临时处理3级类目的匹配问题，之后应该是精确搜索
        category_prefix = self.get_category_prefix(category, level="L3", temp=temp_cut)

        # 调用地图API进行搜索，TODO 搜索参数日后可以由LLM控制
        radius=1000
        for _ in range(3):
            poi_list = super().search(
                center=center,
                radius=radius,
                category_prefix=category_prefix,
                limit=10
                )
            if len(poi_list) > 0:
                break
            radius=radius+500
        poi_list = [p[0] for p in poi_list]
        # print(poi_list)

        # 增加随机扰动，以增加从搜索结果中得到核心目标的难度
        if shuffle:
            for shuf_id in random.sample(list(self.supported_poi_ids.keys()), 3):
                if shuf_id[:-1] != category_prefix:
                    poi_list_shuf = super().search(
                        center=center,
                        radius=1000,
                        category_prefix=shuf_id[:-1],
                        limit=3
                    )
                    if len(poi_list_shuf) > 0:
                        poi_list = poi_list + [p[0] for p in poi_list_shuf]
                        break
            random.shuffle(poi_list)
            poi_list = poi_list[:min(len(poi_list), 10)]
        
        if clean:
            # 校验并清理POI返回值
            poi_list_clean = []
            for info in poi_list:
                poi = info
                # 不在实验范围内的去除
                if not self.check_in_region(poi["shapely_lnglat"]):
                    continue
                poi_slim = {}
                for item in poi:
                    if item in ["id", "category", "aoi_id"]:
                        poi_slim[item] = poi[item]
                poi_slim["name"] = self.LANGUAGE.get_poi_name(poi["id"], self._city_map)
                poi_list_clean.append(poi_slim)
            # print(poi_list_clean)
        else:
            poi_list_clean = poi_list
        
        # 记录见过的POI并进行局部编码
        self.update_pois_in_visiable_dict(poi_list_clean)
        self.update_current_step_visiable_pois(poi_list_clean)

        return poi_list_clean

    # 将search返回的poi列表组织成周围环境的描述
    def search_or_look_res_to_text(self, poi_list: list, action:str):
        if len(poi_list) == 0:
            return "There are no POIs nearby. Nothing happens."

        # 这里只给出第三级类目代表的通用POI，就像virtual-home中的物品一样
        info = "There are nearest {} POIs aroud you, they are:".format(len(poi_list))
        info = info + ",".join(self.get_current_step_visiable_pois(action=action))

        return info

    ###########################
    # Action 2: Navigate
    # TODO 后续需要配合引入更多交通方式，并增加交通执行的细节
    # 问题：由于实时获取信息，人工静态描述很难实现，需要引入实时反馈机制，可能可以依赖GPT4+全局信息来补充细节
    async def navigate(self, text_local_id):
        if text_local_id not in self.get_current_step_visiable_pois(action=Action.SEARCH.value):
            return None, None

        # poi_local_id翻译回global_id
        poi_id = self.from_local_to_global(text_local_id)

        # 查询POI所在AOI信息
        aoi_id = self.get_aoi_of_poi(poi_id)
        walk_route = await self.get_walking_route(aoi_id)
        drive_route = await self.get_driving_route(aoi_id)
        
        # TODO 这里只是简单近似，提供金钱成本
        if walk_route is not None:
            walk_route["price"] = 0
        if drive_route is not None and "eta" in drive_route:
            drive_route["price"] = drive_route["eta"]*10

        return walk_route, drive_route
    
    async def navigate_all(self, text_local_id_list):
        routes= {}
        for local_id in text_local_id_list:
            routes[local_id] = await self.navigate(local_id)
        return routes

    # 将navigate返回的路径信息组织成文本描述
    def navigate_to_text(self, text_local_id, walk_route, drive_route, simplify_route=True):
        
        if walk_route is None:
            info_walk = "you cannot walk to {}".format(text_local_id)
        else:
            info_walk = "{'walk': {'routes':%s, 'time_cost':%d, 'price_cost':%d}}" % (
                "" if simplify_route else walk_route["routes"],
                walk_route["eta"],
                walk_route["price"]
            )
        
        if drive_route is None:
            info_drive = "you cannot drive to {}".format(text_local_id)
        else:
            info_drive = "{'drive': {'routes':%s, 'time_cost':%d, 'price_cost':%d}}" % (
                "" if simplify_route else drive_route["routes"], 
                drive_route["eta"],
                drive_route["price"]
            )

        info = "You have two ways to arrive at {}:\n1.{}\n2.{}".format(
            text_local_id, info_drive, info_walk
        )
        return info


    def navigate_all_to_text(self, routes: dict, simplify_route=True):
        info = []
        for local_id in routes:
            if routes[local_id] is None:
                continue
            info.append(self.navigate_to_text(local_id, routes[local_id][0], routes[local_id][1], simplify_route=simplify_route))
        if len(info) == 0:
            return "Routing fails, no routes are available." 
        else:
            return "\n".join(info)

    ######################
    # 增加一步一步的移动操作，近似模拟CityWalk行为，通过闲逛收集信息实现
    async def navigate_to_poi(self, id, mode="drive"):

        # 同一个aoi_id内的poi共享一样的导航路径信息，隐含AOI中包含哪些POI的信息

        poi_id = id
        poi_info = self._city_map.get_poi(poi_id)
        aoi_id = poi_info["aoi_id"]
        poi_name = self.LANGUAGE.get_poi_name(poi_id, self._city_map)
        start_poi_name = self.LANGUAGE.get_poi_name(self.init_poi_id, self._city_map)
        poi_addr = self.LANGUAGE.get_poi_address(poi_id, self._city_map)
        start_poi_addr = self.LANGUAGE.get_poi_address(self.init_poi_id)
        if start_poi_addr == "":
            start_poi_addr_str = ""
        else:
            start_poi_addr_str = "({})".format(start_poi_addr)
        if poi_addr == "":
            poi_addr_str = ""
        else:
            poi_addr_str = "({})".format(poi_addr)

        if aoi_id is None:
            return None, "POI is not belong to any AOI. " + NavigateStatus.NO_ROUTE

        if mode == "drive":
            route = await self.get_driving_route(aoi_id)
        elif mode == "walk":
            route = await self.get_walking_route(aoi_id)
        else:
            raise NotImplementedError
        
        road_list = []
        junction_list = []
        if mode == "drive":
            for road_id in route["road_ids"]:
                road_info = self._city_map.get_road(road_id)
                lane_info = self._city_map.get_lane(road_info["lane_ids"][0])
                road_list = self.road_info_collect(road_info, lane_info, road_list)
        elif mode == "walk":
            for lane in route["route"]:
                lane_id = lane["lane_id"]
                lane_info = self._city_map.get_lane(lane_id)
                road_info = self._city_map.get_road(lane_info["parent_id"])
                road_list = self.road_info_collect(road_info, lane_info, road_list)
        else:
            raise NotImplementedError
        
        self.current_road_list = copy.deepcopy(road_list)
        self.routing_road_list = copy.deepcopy(road_list)
        # TODO 暂不考虑合并同名道路，后续再考虑，文本模板待更新
        info_text = start_dest_text(start_poi_name, start_poi_addr_str, poi_name, poi_addr_str)
        infos = []
        files = []
        for i in range(len(road_list)):
            current_road = road_list[i]
            infos.append(describe_one_step_text(int(current_road[1]/100)*100, current_road[0], current_road[3]))
            # 增加该步poi信息
            lane_info_step = self._city_map.get_lane(current_road[2])
            endpoint_lnglat = lane_info_step["shapely_lnglat"].coords[-1]
            endpoint_xy = lane_info_step["shapely_xy"].coords[-1]

            # update position
            self.position = Position(
                xy_position=XYPosition(x=endpoint_xy[0], y=endpoint_xy[1]),
                longlat_position=LongLatPosition(longitude=endpoint_lnglat[0], latitude=endpoint_lnglat[1]),
            )

            if LANDMARK_DATA == True:
                file_name, interest_step_text = self.describe_step_interests(current_road[2], current_road[3])
                if interest_step_text is not None:
                    infos.append(interest_step_text)
                if file_name is not None:
                    files.append(file_name)

            if i < len(road_list) - 1:
                next_road = road_list[i + 1]
                basic_direction = next_road[3].split(" ")[-1]
                junction_name = junc_name_text(current_road[0], next_road[0])
                junction_list.append(junction_name)
                infos.append(junc_walk_text(basic_direction, junction_name))
        
        self.routing_junction_list = copy.deepcopy(junction_list)
        if EVAL_DATA == True:
            info_text['routes'].extend(infos)
            json_info_text = json.dumps(info_text, ensure_ascii=False)
            return road_list, json_info_text, files
        else:
            info_text = info_text + ", ".join(infos)
            return road_list, info_text, files
        

    def describe_step_interests(self, lane_id, direction):
        """描述每一步看到的POI/AOI"""
        if VISION_DATA == True:
            road_id = self._city_map.get_lane(lane_id)["parent_id"]
            landmark_file = os.path.join(RESOURCE_PATH, "match_combined_landmarks_{}.csv".format(REGION_EXP))
            landmark_df = pd.read_csv(landmark_file)
            matching_rows = landmark_df[landmark_df['road_id'] == road_id]
            if matching_rows.empty:
                match_flag = False
            else:
                # 优先选择 landmark 非空的行
                non_empty_landmark_rows = matching_rows[matching_rows['landmark'].notna() & (matching_rows['landmark'] != '')]

                # 如果存在 landmark 非空的行，选择这些行中的第一个
                if not non_empty_landmark_rows.empty:
                    selected_row = non_empty_landmark_rows.iloc[0]
                    match_flag = True
                else:          
                    match_flag = False
        pos = self.get_position()
        center = (pos["xy_position"]["x"], pos["xy_position"]["y"])
        # print(f"Received center: {center}") 

        radius = self.nearby_params["radius"]
        # 最多返回的poi数量
        limit = 10
        category_supported = category_mapping()

        # 用于存储所有分类的POI和距离
        all_interests = [] 
        for category_prefix in category_supported.keys():
            # print(f"Category prefix: {category_prefix}")
            interest_list = self._city_map.query_pois(center, radius, category_prefix, limit)
            # print(f"Interest list: {interest_list}")    
            filtered_interests = [interest for interest in interest_list if 'name' in interest[0] and interest[0]['name']]
            # print(f"Filtered interests: {filtered_interests}")
            all_interests.extend(filtered_interests)
        
        all_interests_sorted = sorted(all_interests, key=lambda x: x[1])
        sorted_interests = [p[0] for p in all_interests_sorted]
        # 描述的poi/aoi数量
        describe_number = 3
        # 当前面朝方向向量映射
        direction_vectors = {
            "from east to west": (-1, 0),
            "from west to east": (1, 0),
            "from south to north": (0, 1),
            "from north to south": (0, -1),
            "from southeast to northwest": (-np.sqrt(2)/2, np.sqrt(2)/2),
            "from northwest to southeast": (np.sqrt(2)/2, -np.sqrt(2)/2),
            "from southwest to northeast": (np.sqrt(2)/2, np.sqrt(2)/2),
            "from northeast to southwest": (-np.sqrt(2)/2, -np.sqrt(2)/2),
        }
        direction_vector = direction_vectors[direction]
        interests_side = {'left': [], 'right': [], 'on the line': []}
        for interest in sorted_interests[:describe_number]:
            interest_pos = interest['shapely_xy']
            vector_to_point = (interest_pos.x - center[0], interest_pos.y - center[1])
            # 计算叉积
            cross_product = np.cross(direction_vector, vector_to_point)
            if cross_product > 0:
                side = "left"
            elif cross_product < 0:
                side = "right"
            else:
                side = "on the line"

            interests_side[side].append(interest)
        interest_text = step_interests_text(interests_side)
        if VISION_DATA == True:
            file_name = None
            if match_flag == True:
                file_name = selected_row['file_name']
                landmark = selected_row['landmark']
                # 保持句子连贯性
                if isinstance(landmark, str):  
                    landmark = landmark.rstrip('.')
                if interest_text != None:
                    interest_text = interest_text + "<image> " + landmark
                else:
                    interest_text = "<image> " + landmark
            return file_name, interest_text
        else:
            return None, interest_text
        

    def road_info_collect(self, road_info, lane_info, road_list):
        road_length = lane_info["length"]
        lane_id = lane_info["id"]
        road_id = road_info["id"]
        startpoint_xy = lane_info["shapely_xy"].coords[0]
        endpoint_xy = lane_info["shapely_xy"].coords[-1]
        basic_direction, direction = direction_description(cal_angle(Point(startpoint_xy), Point(endpoint_xy)))

        if road_id >=300000000:
            road_name =  "Junction"
        else:
            try:
                road_name = self.LANGUAGE.get_road_name(road_id, self._city_map)
                if not road_name: 
                    road_name = "unknown road"

            except IndexError as e:
                print(e)
                road_name = "unknown road"
        
        if road_list != []:
            last_road = road_list[-1]
            # 判断最后一条道路是否与当前道路相同
            if last_road[0] == road_name and last_road[3] == direction:
                # 如果最后一条道路长度没有达到阈值，或当前道路长度没有达到阈值，则合并
                if last_road[1] < self.min_road_length or road_length < self.min_road_length:
                    last_road[1] += road_length  
                    last_road[2] = lane_id  
                else:
                    # 如果最后一条道路长度已达到阈值，且当前道路长度大于或等于阈值，则作为新条目添加
                    road_list.append([road_name, road_length, lane_id, direction, "lane"])
            else:
                # 如果最后一条道路与当前道路不同
                if road_list[-1][1] < self.min_road_length:
                    # 如果最后一条道路的长度小于最小要求，则移除
                    road_list.pop()
                # 添加新条目
                road_list.append([road_name, road_length, lane_id, direction, "lane"])
        else:
            # 如果列表为空，直接添加新条目
            road_list.append([road_name, road_length, lane_id, direction, "lane"])
        
        # 检查列表中最后一条道路的长度
        if road_list and road_list[-1][1] < self.min_road_length:
            road_list.pop()

        return road_list


    async def move_step_by_step(self, id, mode="drive"):
        """执行单步移动操作，drive和walk，并返回移动路线及移动后所在位置看到的POI信息"""
        # 没有导航路径
        if len(self.routing_road_list)==0:
            return (True, NavigateStatus.NO_ROUTE.value, [])
        
        # 已经走完全部导航路径
        if len(self.current_road_list)==0:
            poi_id = id
            aoi_id = self.get_aoi_of_poi(poi_id)
            # aoi不可得
            if aoi_id == None:
                return (True, NavigateStatus.DES_NONE.value, [])
            
            # 完成移动
            self.init_position(self.init_aoi_id)
            if mode == "drive":
                status = await self.drive_to(aoi_id)
            elif mode == "walk":
                status = await self.walk_to(aoi_id)
            else:
                raise NotImplementedError
            if status:
                try:
                    poi_name = self.LANGUAGE.get_poi_name(poi_id, self._city_map)
                except Exception as e:
                    poi_name = str(poi_id)+"-Unknown POI name"
                return (True, poi_name, [])
            else:
                return (True, NavigateStatus.NAV_FAIL.value, [])
        
        # 按照交通方式选择实际单步移动方法
        road_name, road_length, lane_id, direction, _ = self.current_road_list.pop(0)
        lane_info = self._city_map.get_lane(lane_id)
        endpoint_lnglat = lane_info["shapely_lnglat"].coords[-1]
        endpoint_xy = lane_info["shapely_xy"].coords[-1]

        # update position
        self.position = Position(
            xy_position=XYPosition(x=endpoint_xy[0], y=endpoint_xy[1]),
            longlat_position=LongLatPosition(longitude=endpoint_lnglat[0], latitude=endpoint_lnglat[1]),
        )

        return (False, describe_one_step_text(int(road_length/100)*100, road_name, direction), [])


    def get_nearby_interests(self, detail_interest=False):
        """返回所在位置100m范围内的所有POI/AOI"""
        
        pos = self.get_position()
        center = (pos["xy_position"]["x"], pos["xy_position"]["y"])
        # print(f"Received center: {center}") 

        radius = self.nearby_params["radius"]
        limit = self.nearby_params["limit"]
        # 定义优先关注的POI类别
        category_supported = category_mapping()
        interest_info = {}
        for category_prefix in category_supported.keys():
            interest_list = self._city_map.query_pois(center, radius, category_prefix, limit)
            interest_list = [p[0] for p in interest_list if p[0]['name']]
            interest_info[category_supported[category_prefix]] = interest_list
        
        has_category = self.nearby_params["has_category"]>0
        interest_text = describe_pois_via_text(self._city_map, interest_info, radius, has_category, detail_interest)

        return (interest_info, interest_text)


    def register_aoi_info(self, aoi_id, aoi_name):
        if aoi_name not in self.register_aoi:
            self.register_aoi[aoi_name] = aoi_id
    
    def register_poi_info(self, poi_id, poi_name):
        if poi_name not in self.register_poi:
            self.register_poi[poi_name] = poi_id

    ######################

    #################
    # Action 9: Walk, Drive
    async def walk(self, text_local_id: str):
        # TODO 增加异常检测，处理不在环境内的情况
        if text_local_id not in self.get_current_step_visiable_pois(Action.NAVIGATE.value):
            return False
        
        # poi_local_id翻译回global_id
        poi_id = self.from_local_to_global(text_local_id)

        # 查询POI所在AOI信息
        aoi_id = self.get_aoi_of_poi(poi_id)

        self.update_current_step_visiable_pois([])

        return await self.walk_to(aoi_id)
    
    def walk_to_text(self, status: bool, text_local_id: str):
        if status == True:
            return "You have arrived at the entrance of the area where {} is located.".format(text_local_id)
        else:
            return "You fail to arrive at {} due to invalid action with nothing happens.".format(text_local_id)

    # Action 10: Drive
    async def drive(self, text_local_id):
        # TODO 增加异常检测，处理不在环境内的情况
        if text_local_id not in self.get_current_step_visiable_pois(Action.NAVIGATE.value):
            return False

        # poi_local_id翻译回global_id
        poi_id = self.from_local_to_global(text_local_id)

        # 查询POI所在AOI信息
        aoi_id = self.get_aoi_of_poi(poi_id)

        self.update_current_step_visiable_pois([])

        return await self.drive_to(aoi_id)
    
    def drive_to_text(self, status: bool, text_local_id: str):
        if status == True:
            return "You have arrived at the entrance of the area where {} is located".format(text_local_id)
        else:
            return "You fail to arrive at {} due to invalid action with nothing happens.".format(text_local_id)
    
    ############################
    # Action 11: LookIndoor 建筑内部行为
    def look(self, limit=10):
        poi_list = self.look_indoor(limit=limit)
        # 记录见过的POI并进行局部编码

        self.update_pois_in_visiable_dict(poi_list)
        self.update_current_step_visiable_pois(poi_list)

        return poi_list

    ##########################
    # Aciton 12: MoveIndoor
    def move(self, text_local_id):
        # TODO 增加异常检测，处理不在环境内的情况
        if text_local_id not in self.get_current_step_visiable_pois(Action.MOVE_IN_DOOR.value):
            return False

        poi_id = self.from_local_to_global(text_local_id)
        
        try:
            status = self.move_indoor(poi_id)
        except ValueError as e:
            print(e)
            status = False

        return status
    
    def move_to_text(self, status: bool, text_local_id: str):
        if status:
            return "You have entered {}".format(text_local_id)
        else:
            return "You fail to move to {} due to {}".format(text_local_id, "you do not konw the way to it. Nothing happens.")

    #############################
    # 找不到POI时，随机游走一下
    def random_walk(self):
        current_nearby_pois = self.get_current_step_visiable_pois(action=Action.EXPLORE.value)
        if len(current_nearby_pois)==0:
            return False, None
        else:
            text_local_id = random.choice(current_nearby_pois)
            return self.move(text_local_id), text_local_id
    

    def random_walk_to_text(self, status: bool, text_local_id: str):
        if status:
            return "You arrived at {} after random walking".format(text_local_id)
        else:
            return "You cannot random walk. Nothing happens."


    ###############################################3
    # 模拟器对外接口 gym API
    # 按照给定动作执行，并更新环境状态
    async def step(self, action_object, run_mode=RunMode.NORMAL.value, detail_interest=False):
        # last_time = time.time()

        # Runing Mode 1: citywalk，在城市内闲逛，获取各种POI分布信息
        if run_mode == RunMode.CITY_WALK.value:
            script_action, script_object = action_object.split(" ", 1)
            self.script_action = script_action
            self.script_object = script_object

            if script_action == Action.NAVIGATE.value:
                observation = await self.navigate_to_poi(id=self.register_poi[script_object], mode="drive")
            elif script_action in [Action.WALK.value, Action.DRIVE.value]:
                observation = await self.move_step_by_step(id=self.register_poi[script_object], mode="drive")
            else:
                raise NotImplementedError
            observation = self.get_observation(observation, run_mode, detail_interest)
            reward, done, info = self.get_reward(observation["observations"], run_mode)

            return observation, reward, done, info
        
    
    # 获取当前环境状态，原始形式以及文本形式
    # agent自己的状态改变是主要的环境，环境本身不太有什么变化，
    def get_observation(self, observation, run_mode=RunMode.NORMAL.value, detail_interest=False):
        obs = {"position": self.get_position()}
        obs_text = ""

        # Runing Mode 1
        if run_mode == RunMode.CITY_WALK.value:
            if self.script_action == Action.NAVIGATE.value:
                # self.navigate_to_poi()
                obs["routes"] = observation[0]
                obs_text = observation[1]
            elif self.script_action in [Action.WALK.value, Action.DRIVE.value]:
                # self.move_step_by_step()
                obs["status"] = observation[0]
                interests, interests_text = self.get_nearby_interests(detail_interest)
                obs["surroundings"] = interests

                # 获取当前位置的AOI信息
                current_aoi_id = self.position.aoi_position.aoi_id
                current_aoi_name = self.LANGUAGE.get_aoi_name(current_aoi_id, self._city_map)

                if self.routing_junction_list:
                    junction_name = self.routing_junction_list.pop(0)
                else:
                    junction_name = ""
                # grid_x, grid_y = lnglat2grid(self.region_center, (obs["position"]["longlat_position"]["longitude"], obs["position"]["longlat_position"]["latitude"]))
                longitude = round(obs["position"]["longlat_position"]["longitude"], 4)
                latitude = round(obs["position"]["longlat_position"]["latitude"], 4)

                
                if EVAL_DATA == True:
                    if observation[0] == False:
                        position_text= current_aoi_position(junction_name, longitude, latitude, current_aoi_name)
                    else:
                        position_text = final_position(observation[1], longitude, latitude, current_aoi_name)
                    position_text = json.dumps(position_text, ensure_ascii=False)
                    interests_text = json.dumps(interests_text, ensure_ascii=False)
                    obs_description= json.dumps(observation[1], ensure_ascii=False)
                    if observation[0] == False:
                        obs_text = obs_description + "\n" + position_text + "\n" + interests_text
                    else:
                        obs_text = position_text + "\n" + interests_text
                else:
                    if observation[0] == False:
                        position_text= current_aoi_position(junction_name, longitude, latitude, current_aoi_name)
                        obs_text = observation[1] + "\n" + position_text + " \n" + interests_text
                    else:
                        position_text = end_point_text(observation[1])
                        obs_text = position_text + " \n" + interests_text

            else:
                raise NotImplementedError
            return {"observations": obs, "observations_text": obs_text, "files": observation[2]}


        # Runing Mode 2
        if self.script_action in [Action.SEARCH.value, Action.LOOK_IN_DOOR.value]:
            obs["surroundings"] = observation
            obs_text = self.search_or_look_res_to_text(obs["surroundings"], action=self.script_action)
        elif self.script_action in [Action.WALK, Action.DRIVE, Action.MOVE_IN_DOOR, Action.DRINK, Action.BUY, Action.EAT]:
            obs["status"] = observation
            if self.script_action == Action.WALK.value:
                obs_text = self.walk_to_text(obs["status"], self.script_object)
            elif self.script_action == Action.DRIVE.value:
                obs_text = self.drive_to_text(obs["status"], self.script_object)
            elif self.script_action == Action.MOVE_IN_DOOR.value:
                obs_text = self.move_to_text(obs["status"], self.script_object)
            elif self.script_action in [Action.DRINK.value, Action.BUY.value, Action.EAT.value]:
                obs_text = self.do_service_to_text(obs["status"], self.script_object, self.script_action)
        elif self.script_action in [Action.EXPLORE.value]:
            obs["status"] = observation[0]
            self.script_object = observation[1]
            obs_text = self.random_walk_to_text(obs["status"], self.script_object)
        elif self.script_action in [Action.NAVIGATE.value]:
            obs["routes"] = observation
            obs_text = self.navigate_all_to_text(obs["routes"], simplify_route=True)

        return {"observations": obs, "observations_text": obs_text}
    
    def get_reward(self, observation, run_mode=RunMode.NORMAL.value):
        # Runing Mode CityWalk
        if run_mode == RunMode.CITY_WALK.value:
            if self.script_action in [Action.WALK.value, Action.DRIVE.value]:
                if observation["status"]:
                    done = True
                    info = {"finished": done, "reason": "Arrive at the destination"}
                else:
                    done = False
                    info = {"finished": done, "reason": "running"}
            else:
                done = False
                info = {"finished": done, "reason": "running"}
            return {}, done, info


    # 获取原始动作+所有object构成的动作空间, e.g., search foods，目前实现效率较低，需要改进，不需要一定可执行
    def get_action_space(self, init=False, run_mode=RunMode.NORMAL.value, goal=""):
        # 基础动作空间
        if init:
            return [a.value for a in Action]
        
        if run_mode == RunMode.CITY_WALK.value:
            return [Action.NAVIGATE.value+" {}".format(goal), Action.WALK.value+" {}".format(goal)]
        

    # 需要时刻与init函数对齐
    def reset(self):
        aoi = self._city_map.get_aoi(self.init_aoi_id)
        if aoi is None:
            raise ValueError(f"aoi {self.init_aoi_id} not found")
        xy = cast(Polygon, aoi["shapely_xy"]).centroid
        x, y = xy.coords[0][0], xy.coords[0][1]
        lng, lat = self._city_map.xy2lnglat(x, y)

        self.position = Position(
            aoi_position=AoiPosition(aoi_id=self.init_aoi_id),
            xy_position=XYPosition(x=x, y=y),
            longlat_position=LongLatPosition(longitude=lng, latitude=lat),
        )  # 当前位置

        self.visiable_pois, self.visiable_pois_revserse = {}, {}
        self.visiable_pois_global = {}

        self.time_cost = 0  # 总时间代价
        self.totlal_steps = 0
        self.price_cost = 0

        # TODO 如何定义任务目标
        self.current_step_visiable_pois_local_view = []
        self.current_step_known_pois_local_view = []

        self.current_road_list = []
        self.routing_road_list = []

        self.script_object = None
        self.script_action = None

        return
