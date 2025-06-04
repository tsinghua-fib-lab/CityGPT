#!/usr/bin/bash
set -x
set -e

# 0.读取fsq的poi并写到本地
python3 from_fsq_to_raw_pois.py

# 1.挂梯子获得osm的路网 AOI POI数据
python3 net_aoi_poi.py

# 2.build地图并写入mongodb
python3 build_full_map.py --workers 13

# 3.在mongodb基础上原地添加nearby的aoi名称
python3 add_all_aoi_names.py

# 4.替换fsq的poi并写入mongodb
python3 replace_all_pois.py

# 5.打印所有地图
python3 print_llmmap_names.py
