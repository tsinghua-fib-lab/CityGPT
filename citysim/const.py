
CITY_TO_AGENT_NUM = {
    "shanghai": 100_0000,
    "beijing": 100_0000,
    "newyork": 100_0000,
    "paris": 100_0000,
}
DIS_RATIO = 0.6
RULE_RATIO = 0.8
CITY_TO_TRIP_PB = {
    "shanghai": "moss.trip_china_shanghai_aigc.pb",
    "beijing": "moss.trip_china_beijing_aigc.pb",
    "newyork": "moss.trip_us_newyork_aigc.pb",
    "paris": "moss.trip_france_paris_aigc.pb",
}

conf = {}
conf['num_hiddens'] = 64
conf['num_layers'] = 2
conf['output_noise'] = False
conf['rand_prior'] = True
conf['verbose'] = False
conf['l1'] = 3e-3
conf['lr'] = 3e-2
conf['num_epochs'] = 100
BO_CONF = conf
CITY_TO_MAP_PB = {
    "shanghai": "moss.map_china_shanghai.pb",
    "beijing": "moss.map_china_beijing.pb",
    "newyork": "moss.map_us_newyork.pb",
    "paris": "moss.map_france_paris.pb",
}
def get_host(ITER_TYPE):
    CITY_TO_HOST  = {}
    if ITER_TYPE=="RANDOM":
        CITY_TO_HOST = {
        "shanghai": "localhost:52101",
        "beijing": "localhost:52102",
        "newyork": "localhost:52103",
        "paris": "localhost:52104",
        }
    elif ITER_TYPE=="RULE_BASED":
        CITY_TO_HOST = {
            "shanghai": "localhost:52201",
            "beijing": "localhost:52202",
            "newyork": "localhost:52203",
            "paris": "localhost:52204",
        }
    elif ITER_TYPE=="DO_NOTHING":
        CITY_TO_HOST = {
            "shanghai": "localhost:52401",
            "beijing": "localhost:53126",
            "newyork": "localhost:53001",
            "paris": "localhost:53000",
            "mumbai": "localhost:53128",
            "london": "localhost:53146",
            "moscow": "localhost:53147",
            "tokyo": "localhost:53138",
            "london": "localhost:53139",
            "san_francisco": "localhost:53140",
            "sao_paulo": "localhost:53141",
            "nairobi": "localhost:53142",
            "sydney": "localhost:53143",
        }
    elif ITER_TYPE == "HEBO":
        CITY_TO_HOST = {
            "shanghai": "localhost:52501",
            "beijing": "localhost:52502",
            "newyork": "localhost:52503",
            "paris": "localhost:52504",
    }
    return CITY_TO_HOST
