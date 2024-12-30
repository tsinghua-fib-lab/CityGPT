######### 本文件用于设置全局参数
#### 以下是公用配置
PROXY = "http://127.0.0.1:10190"
# 用于获取osm地址
SERVING_IP = ""
# 本机服务器
SERVER_IP = ""
# 本机部署模型key
LOCAL_MODEL_KEY = ""
############# TODO 不可对外暴露，仅用于内部测试
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
    "wudaokou_small":"map_beijing5ring_withpoi_0424",
    "wudaokou_large":"map_beijing5ring_withpoi_0424",
    "yuyuantan":"map_beijing5ring_withpoi_0424",
    "dahongmen":"map_beijing5ring_withpoi_0424",
    "wangjing":"map_beijing5ring_withpoi_0424",
    "beijing":"map_beijing5ring_withpoi_0424",
    "paris":"map_paris_20240512",
    "newyork":"map_newyork_20240512"
}
# 城市地图端口号
MAP_PORT_DICT={
    "wudaokou_small":54319,
    "wudaokou_large":54319,
    "yuyuantan":54319,
    "dahongmen":54319,
    "wangjing":54319,
    "beijing":54319,
    "paris":54320,
    "newyork":54321
}

REGION_BOUNDARY = {
    # 左下角：116.293669, 39.99254, 右上角：116.359928,40.01967
    # small wudaokou
    "wudaokou_small": [(116.293669, 39.99254), (116.359928, 39.99254), (116.359928,40.01967), (116.293669, 40.01967), (116.293669, 39.99254)],
    "wudaokou_large": [(116.26, 39.96), (116.40,39.96), (116.40, 40.03), (116.26, 40.03), (116.26, 39.96)],
    # 116.447387,39.986287；116.504592,40.020057
    "wangjing": [(116.447387, 39.986287), (116.504592, 39.986287), (116.504592, 40.020057), (116.447387, 40.020057), (116.447387, 39.986287)],
    # 左下角：116.37778,39.838326，右上角：116.450219,39.863752
    "dahongmen": [(116.37778, 39.838326), (116.450219, 39.838326), (116.450219, 39.863752), (116.37778, 39.863752), (116.37778, 39.838326)],
    # 左下角：116.287375,39.908043，右上角：116.368366,39.942128
    "yuyuantan": [(116.287375, 39.908043), (116.368366, 39.908043), (116.368366, 39.942128), (116.287375, 39.942128), (116.287375, 39.908043)],
    # 左下角：2.2493, 48.8115，右上角：2.4239, 48.9038
    "paris": [(2.2493, 48.8115), (2.4239, 48.8115), (2.4239, 48.9038), (2.2493, 48.9038), (2.2493, 48.8115)],
    # 左下角：-74.0128, 40.7028，右上角：-73.9445, 40.7314
    "newyork": [(-74.0128, 40.7028), (-73.9445, 40.7028), (-73.9445, 40.7314), (-74.0128, 40.7314), (-74.0128, 40.7028)],
    # 左下角：116.1603, 39.6916，右上角：116.6506, 40.083
    "beijing": [(116.1603, 39.6916), (116.6506, 39.6916), (116.6506, 40.083), (116.1603, 40.083), (116.1603, 39.6916)],
}

#### 以下配置用于生成训练数据，simulate文件夹下
# 地图是否为OSM地图，如果区域为beijing，设置为False；其他区域设置为True
OSM_REGION = True
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
        "aoi2coor": "aoi2coor.csv",
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
    # "GPT4o":"gpt-4o"
}


