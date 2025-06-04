# 数据组成说明
当前地图数据由OSM和FSQ共同构成，其中OSM提供路网等数据结构，FSQ提供POI数据作为OSM中POI数据的替换，citymap代码主要描述了如何从城市名字作为输入完成整个地图数据的准备工作

# 关键参数配置
主要是map_config中
- MONGODB_URI # 用来存储地图数据的mongodb数据库地址，需要参考最新数据来，存储在map_config中，
- 城市名字参数确认：[搜索城市名称](../assets/osm-map-p1.png),[搜索结果](../assets/osm-map-p1.png),[Wikipedia字段](../assets/osm-map-p3.png)
- 城市对应经纬度范围的 `all_bbox`
- fsq原始数据路径 `FSQ_PATH`


# 相关代码说明
- 脚本 map_run.sh 是整个运行流程
- 依赖 requirements.txt
- 参数配置 map_config.py
