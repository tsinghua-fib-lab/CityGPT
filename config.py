######### 本文件用于设置全局参数
#### 以下是公用配置
PROXY = "http://127.0.0.1:10190"
# 用于获取osm地址
SERVING_IP = "xxx"
# 本机服务器
SERVER_IP = "xxx"
# 本机部署模型key
LOCAL_MODEL_KEY = "xxx"
#############
MONGODB_URI = ""
###########
MAP_CACHE_PATH="simulate/examples/cache/"
ROUTING_PATH="resource/routing_linux_amd64"
RESOURCE_PATH = "resource/"
# 设置地区
REGION_EXP = "newyork"
# 最短路径长度
MIN_ROAD_LENGTH = 100
# 城市地图信息
MAP_DICT={
    "Beijing":"map_beijing_fsq_20250105",
    "London":"map_london_fsq_20250107",
    "Paris":"map_paris_fsq_20250105",
    "NewYork":"map_newyork_fsq_20250106",
    "SanFrancisco":"map_san_francisco_fsq_20250105"
}
# 城市地图端口号
MAP_PORT_DICT={
    "NewYork":54332,
    "SanFrancisco":54327,
    "London":54325,
    "Paris":54320,
    "Beijing":54319
}

REGION_BOUNDARY = {
    # 左下角：116.293669, 39.99254, 右上角：116.359928,40.01967
    "NewYork": [(-74.255591, 40.496134), (-73.72621, 40.496134), (-73.72621, 40.91481), (-74.255591, 40.91481), (-74.255591, 40.496134)],
    "Beijing": [(115.613909, 39.592447), (117.43, 39.592447), (117.43, 40.31976), (115.613909, 40.31976), (115.613909, 39.592447)],
    "Paris": [(2.224225, 48.8156), (2.4688, 48.8156), (2.4688, 48.89652), (2.224225, 48.89652), (2.224225, 48.8156)],
    "London": [(-0.510375, 51.28676), (0.314881, 51.28676), (0.314881, 51.6828), (-0.510375, 51.6828), (-0.510375, 51.28676)],
    "SanFrancisco": [(-123.173825, 37.63983), (-122.29246797, 37.63983), (-122.29246797, 37.9134296), (-123.173825, 37.9134296), (-123.173825, 37.63983)],
}

#### 以下配置用于生成训练数据，simulate文件夹下
# 设置是否使用OSM地址
OSM_ADDRESS = False
# 设置数据版本
DATA_VERSION = "eval"

# 参数决定生成数据形式，为True时生成结构化数据，为评估提供数据；为False时生成模板泛化的CityWalk数据，生成训练数据
EVAL_DATA = True
#### 下面的参数仅在EVAL_DATA = False时生效
# 参数决定CityWalk数据形式，为True时生成landmark形式单轮导航数据，为False时生成原始形式数据
LANDMARK_DATA = False
# LANDMARK_DATA为False时生效，DETAIL_INTEREST = True,输出详细的POI信息；DETAIL_INTEREST = False,输出简略的POI信息
DETAIL_INTEREST = False
# LANDMARK_DATA = True时生效，参数决定是否生成CityWalk-Vision数据
VISION_DATA = False


#### 以下配置用于生成评估数据，evaluate文件夹下
EVAL_TASK_MAPPING_v1 = {
    "city_image": {
        "poi2coor": "poi2coor.csv",
        "road_length": "eval_road_length.csv", 
        "road_od": "eval_road_od.csv",
        "road_link": "eval_road_link.csv",
        "aoi2addr": "aoi2addr.csv",
        "landmark_path": "eval_landmark_path.csv",  
        "boundary_road": "eval_boundary_road.csv", 
        "road_aoi": "eval_road_aoi.csv",
        "aoi_near": "aoi_near.csv"
    },
    "urban_semantics": {
        "landmark_env": "eval_landmark_env.csv",  
        "aoi2type": "aoi2type.csv", 
        "type2aoi": "type2aoi.csv",  
    },
    "spatial_reasoning_route": {
        "aoi2aoi_dir_routine":"aoi2aoi_dir_routine.csv",
        "aoi2rd_dis_routine":"aoi2rd_dis_routine.csv",
        "aoi2rd_dir_routine":"aoi2rd_dir_routine.csv",
        "aoi2aoi_dis_routine":"aoi2aoi_dis_routine.csv",
    },
    "spatial_reasoning_noroute": {
        "aoi2rd_dis_noroutine":"aoi2rd_dis_noroutine.csv",
        "aoi2rd_dir_noroutine":"aoi2rd_dir_noroutine.csv",
        "aoi2aoi_dis_noroutine":"aoi2aoi_dis_noroutine.csv",
        "aoi2aoi_dir_noroutine":"aoi2aoi_dir_noroutine.csv",
    }
}

EVAL_TASK_MAPPING_v2 = {
    "city_image": {
        "aoi_poi": "aoi_poi.csv", 
        "poi_aoi": "poi_aoi.csv",  
        "poi2coor": "poi2coor.csv",
        "poi2addr": "poi2addr.csv",
        "road_length": "road_length.csv", 
        "road_od": "road_od.csv",
        "road_link": "road_link.csv", 
        "road_arrived_pois": "road_arrived_pois.csv",
        "aoi2addr": "aoi2addr.csv",
        "landmark_path": "landmark_path.csv",
        "boundary_road": "boundary_road.csv", 
        "aoi_boundary_poi": "aoi_boundary_poi.csv",
        "districts_poi_type": "districts_poi_type.csv",
    },
    "urban_semantics": {
        "landmark_env": "landmark_env.csv",  
        "aoi_group": "aoi_group.csv", 
        "poi2type": "poi2type.csv",
        "type2poi": "type2poi.csv",
        "aoi2type": "aoi2type.csv", 
        "type2aoi": "type2aoi.csv",  
    },
    "spatial_reasoning_route": {
    "aoi2aoi_dir_routine": "aoi2aoi_dir_routine.csv",
    "poi2poi_dir_routine": "poi2poi_dir_routine.csv",
    "poi2poi_dis_routine": "poi2poi_dis_routine.csv",
    "poi2aoi_dis_routine": "poi2aoi_dis_routine.csv",
    "poi2aoi_dir_routine": "poi2aoi_dir_routine.csv",
    "poi2rd_dis_routine": "poi2rd_dis_routine.csv",
    "poi2rd_dir_routine": "poi2rd_dir_routine.csv",
    "aoi2rd_dis_routine": "aoi2rd_dis_routine.csv",
    "aoi2rd_dir_routine": "aoi2rd_dir_routine.csv",
    "aoi2aoi_dis_routine": "aoi2aoi_dis_routine.csv",
    },
    "spatial_reasoning_noroute": {
    "aoi2aoi_dir_noroutine": "aoi2aoi_dir_noroutine.csv",
    "poi2poi_dir_noroutine": "poi2poi_dir_noroutine.csv",
    "poi2poi_dis_noroutine": "poi2poi_dis_noroutine.csv",
    "poi2aoi_dis_noroutine": "poi2aoi_dis_noroutine.csv",
    "poi2aoi_dir_noroutine": "poi2aoi_dir_noroutine.csv",
    "poi2rd_dis_noroutine": "poi2rd_dis_noroutine.csv",
    "poi2rd_dir_noroutine": "poi2rd_dir_noroutine.csv",
    "aoi2rd_dis_noroutine": "aoi2rd_dis_noroutine.csv",
    "aoi2rd_dir_noroutine": "aoi2rd_dir_noroutine.csv",
    "aoi2aoi_dis_noroutine": "aoi2aoi_dis_noroutine.csv",
    }
}
DIS2CORNER = 50
STEP = 12
REASON_QUES_NUM = 500
TRAIN_DATA_PATH = "simulate/examples/"
LLM_MODELS = [
    "Qwen2-7B", "Qwen2-72B", "Intern2.5-7B", "Intern2.5-20B", 
    "Mistral-7B", "Mixtral-8x22B", "LLama3-8B", "LLama3-70B", "Gemma2-9B", "Gemma2-27B", 
    "DeepSeek-67B", "DeepSeekV2", "GPT3.5-Turbo", "GPT4-Turbo"]

INFER_SERVER = {
    "OpenAI": ["GPT3.5-Turbo", "GPT4-Turbo", "GPT4omini"],
    "DeepInfra": ["Mistral-7B", "Mixtral-8x22B", "LLama3-8B", "LLama3-70B", "Gemma2-9B", "Gemma2-27B", "LLama-3.2-90B", "LLama-3.2-11B"],
    "Siliconflow": ["Qwen2-7B", "Qwen2-72B", "Intern2.5-7B", "Intern2.5-20B", "DeepSeekV2", "Qwen2-VL-72B"],
    "DeepBricks": ["gpt-4o-mini"]
}
LLM_MODEL_MAPPING = {
    "Qwen2-7B":"Qwen/Qwen2-7B-Instruct",
    "Qwen2-72B":"Qwen/Qwen2-72B-Instruct",
    "Intern2.5-7B":"internlm/internlm2_5-7b-chat",
    "Intern2.5-20B":"internlm/internlm2_5-20b-chat",
    "Mistral-7B":"mistralai/Mistral-7B-Instruct-v0.2", 
    "Mixtral-8x22B":"mistralai/Mixtral-8x22B-Instruct-v0.1",
    "LLama3-8B":"meta-llama/Meta-Llama-3-8B-Instruct",
    "LLama3-70B":"meta-llama/Meta-Llama-3-70B-Instruct",
    "Gemma2-9B":"google/gemma-2-9b-it",
    "Gemma2-27B":"google/gemma-2-27b-it",
    "DeepSeekV2":"deepseek-ai/DeepSeek-V2-Chat",
    "GPT3.5-Turbo":"gpt-3.5-turbo-0125",
    "GPT4-Turbo":"gpt-4-turbo-2024-04-09",
    "GPT4omini":"gpt-4o-mini-2024-07-18",
}


