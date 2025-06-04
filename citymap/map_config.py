WORKERS = 13
MONGODB_URI = ""
DB = "srt"

# fsq数据路径
FSQ_PATH = ""
# 输出路径
FSQ_PKL_PATH = "./fsq_pkls"

# OSM被墙需要挂代理
PROXIES = {
    "http": "127.0.0.1:1090",
    "https": "127.0.0.1:1090",
}

# ATTENTION:这个名字去osm官网搜地名 就能找到该字段
city_2_wiki_name = {
    "paris":"fr:Paris",
    "newyork":"en:New York City",
    "beijing":"zh:北京市",
    "london":"en:London",
    "san_francisco":"en:San Francisco",
}

all_bbox = {
    "paris": {
        "min_lon": 2.224225,
        "max_lon": 2.4688,
        "min_lat": 48.8156,
        "max_lat": 48.89652,
    },
    "newyork": {
        "min_lon": -74.255591,
        "max_lon": -73.72621,
        "min_lat": 40.496134,
        "max_lat": 40.914816,
    },
    "beijing": {
        "min_lon": 115.613909,
        "max_lon": 117.43,
        "min_lat": 39.592447,
        "max_lat": 40.31976,
    },
    "london": {
        "min_lon": -0.510375,
        "max_lon": 0.314881,
        "min_lat": 51.28676,
        "max_lat": 51.68288,
    },
    "san_francisco": {
        "min_lon": -123.173825,
        "max_lon": -122.29246797,
        "min_lat": 37.63983,
        "max_lat": 37.9134296,
    }
}
