import hashlib
import json
import logging
import os
import pdb
import pickle
import random
import re
import shutil
import sys
import time
from datetime import datetime, timedelta
from openai import OpenAI #type
import httpx

import numpy as np
import openai
from tqdm import tqdm

MODEL = "gpt-3.5-turbo"

# 关于网格化的几个超参数
XMIN = 435300
XMAX = 457300
YMIN = 4405900
YMAX = 4427900
D = 250  # 决定了网格化的粒度  
N = int((XMAX-XMIN)/D)
    
def XY2Grid_ID(x,y):
    i = (x - XMIN) // D  # 可以直接用这种方式来决定是在哪个里面
    j = (y - YMIN) // D
    id = int(i*N+j)
    return id
    

def printSeq(seq):
    # 其实就是一个把序列逐元素打印的函数
    for item in seq:
        print(item)


def setup_logger(agentid):
    # os.remove("Logs/record.log")
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler("Logs/record_{}.log".format(agentid))

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger,"Logs/record_{}.log".format(agentid)


def askChatGPT(messages, model="gpt-3.5-turbo", temperature = 1):
    response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            temperature = temperature,
        )
    addtoken(response.usage.total_tokens)
    answer = response.choices[0].message["content"]
    return answer.strip()


def askLocalGPT(messages, model, temperature = 1):
    
    # response = openai.chat.completions.create(
    #     model = model_name,
    #     messages = messages,
    #     temperature = temperature,
    #     max_tokens = 2048,)  # which control the randomness of the results, between 0 and 2
    # addtoken(response.usage.total_tokens)
    # answer = response.choices[0].message.content
    # return answer.strip()

    response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            temperature = temperature,
            max_tokens = 2048,
        )
    # addtoken(response.usage.total_tokens)
    answer = response.choices[0].message["content"]
    return answer.strip()


def setOpenAi(keyid = 0):
    return ""

def setLocalOpenAi(choice = 0):
    openai.api_base = "xxx" if choice == 0 else "xxx" 
    openai.api_key = "EMPTY"
    model_name = "ChatGLM3-6B" if choice == 0 else "CityGPT-v5" 
    return model_name


def setMyOpenAi(model):
    if model == "ChatGLM3-6B":
        openai.api_base = "xxxx"
        openai.api_key = "EMPTY"
    elif model == "CityGPT-v5":
        openai.api_base = "xxx"
        openai.api_key = "EMPTY"
    elif model == "CityGPT-v10":
        openai.api_base = "xxx"
        openai.api_key = "EMPTY"
    else:
        # GPT系列model,这个是openai key
        openai.api_key = ""



def get_agent_action_simple(session, model_name="gpt-3.5", temperature=1.0, max_tokens=200):

    # 模型API设置
    if "gpt-3.5" in model_name or "gpt-4" in model_name:
        model_name_map = {
            "gpt-3.5": "gpt-3.5-turbo-0125",
            "gpt-4": "gpt-4-0125-preview"
        }
        model_name = model_name_map[model_name]
        client = OpenAI(
            http_client=httpx.Client(proxy="http://127.0.0.1:10190"),
            api_key=''
            )
    elif "meta-llama" in model_name or "mistralai" in model_name:
        client = OpenAI(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key="",
        http_client=httpx.Client(proxies="http://127.0.0.1:10190"),
            )
    elif "deepseek-chat" in model_name:
        client = OpenAI(
        api_key="",
        base_url="https://api.deepseek.com/v1"
        )
    elif "chatglm3-6B-v21.4:23131" in model_name:
        model_name, port = model_name.split(":")
        client = OpenAI(
            base_url="http://xxx:{}/v1".format(port),
            api_key=""
        ) 
    else:
        model_name, port = model_name.split(":")
        client = OpenAI(
            base_url="http://xxx:{}/v1".format(port),
            api_key=""
        )

    MAX_RETRIES = 1
    WAIT_TIME = 1
    for i in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                        model=model_name,
                        messages=session,
                        temperature=temperature,
                        max_tokens=100,
                    )
            return response.choices[0].message.content
        except Exception as e:
            if i < MAX_RETRIES - 1:
                time.sleep(WAIT_TIME)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "OpenAI API Error."


    

                

def getTime(timestr):
    timelist = timestr[1:-1].split(',')
    return [int(timelist[0]), int(timelist[1])]

def printQA(Q, A, logger, additional_info = ''):
    logger.info(additional_info + 'Question: {}'.format(Q))
    logger.info(additional_info + 'Answer: {}'.format(A+'\n'))
    
def addtoken(num):
    try:
        with open("tokens.txt", "r") as f:  # 打开文件
            data = f.read()  # 读取文件
            nownum = int(data)        
            
        if num == -1:
            nownum = 0
        else:
            nownum = nownum + num
        
        with open("tokens.txt","w+") as f:
            f.write(str(nownum))  # 自带文件关闭功能，不需要再写f.close()
    except:
        pass
    
    # print('add tokens num', num)
    
def timeEval(dis):
    # 距离单位是米
    # TODO：根据距离得到路程所需的时间
    # 目前只是一个非常简陋的时间估计器
    return int(dis/70)  # 时间评估的结果只返回一个整数,这其实是一个比较简陋的估计

def calTime(start, duration):
    h1, m1  = start.split(':')
    h1 = int(h1)
    m1 = int(m1)
    m1 = m1 + duration
    if m1 > 59:
        h1 += 1
        m1 -= 60
        if h1 > 23:
            h1 = h1-24
    h1 = str(h1)
    m1 = str(m1)
    if len(m1) == 1:
        m1 = '0' + m1
    return str(h1) + ':' + str(m1)
        
def turntime2list(time):
    time = time[1:-1]
    time = time.split(',')
    time = [i.strip() for i in time]
    return (time[0], time[1])

def getDirectEventID(event):
    # 直接从event查询具体的POI, 也不再问具体的类目
    # ["go to work", "go home", "eat", "do shopping", "do sports", "excursion", "leisure or entertainment", "go to sleep", "medical treatment", "handle the trivialities of life", "banking and financial services", "cultural institutions and events"]: 
    if event in ['have breakfast', 'have lunch', 'have dinner', 'eat']:
        return "10"
    elif event == 'do shopping':
        return "13"
    elif event == 'do sports':
        return "18"
    elif event == 'excursion':  # 这里指短期的旅游景点
        return "22"
    elif event == 'leisure or entertainment':
        return "16"
    elif event == 'medical treatment':
        return "20"
    elif event == 'handle the trivialities of life':
        return "14"
    elif event == 'banking and financial services':
        return "25"
    elif event == 'government and political services':
        return "12"
    elif event == 'cultural institutions and events':
        return "23"
    else:
        print(event)
        print('\nIn function event2cate: The selected choice is not in the range!\n')
        sys.exit(0)
    
    
def add_time(start_time, minutes):
    """
    计算结束后的时间，给出起始时间和增量时间（分钟）
    :param start_time: 起始时间，格式为 '%H:%M'
    :param minutes: 增量时间（分钟）
    :return: 结束后的时间，格式为 '%H:%M'；是否跨越了一天的标志
    """
    # 将字符串转换为 datetime 对象，日期部分设为一个固定的日期
    start_datetime = datetime.strptime('2000-01-01 ' + start_time, '%Y-%m-%d %H:%M')

    # 增加指定的分钟数
    end_datetime = start_datetime + timedelta(minutes=minutes)

    # 判断是否跨越了一天
    if end_datetime.day != start_datetime.day:
        cross_day = True
    else:
        cross_day = False

    # 将结果格式化为字符串，只包含时间部分
    end_time = end_datetime.strftime('%H:%M')

    return end_time, cross_day

def getTimeFromZone(timeZone0):
    time0, time1 = timeZone0.split('-')
    time0 = float(time0)/2  # 这里已经化成小时了
    time1 = float(time1)/2
    # print(time0)
    # print(time1)
    
    sampleResult = random.uniform(time0, time1)  # 采样一个具体的时间值出来,单位是小时
    # print(sampleResult)  
    minutes = int(sampleResult*60)
    return minutes

def getTimeDistribution():
    file = open('Data/timeDistribution_Event.pkl','rb') 
    timeDistribution_Event = pickle.load(file)
    timeDistribution_Event['banking and financial services'] = timeDistribution_Event['handle the trivialities of life']
    timeDistribution_Event['cultural institutions and events'] = timeDistribution_Event['handle the trivialities of life']
    return timeDistribution_Event


def extract_single_number(text):
    # 使用正则表达式匹配第一个数字
    match = re.search(r'\d+', text)
    if match:
        # 返回匹配到的数字
        return int(match.group())
    else:
        # 如果没有找到数字，则返回 None 或者其他适当的值
        return -1

def extract_longest_number(s):
    numbers = re.findall(r'\d+', s)
    if not numbers:
        return None
    longest_number = max(numbers, key=len)
    return int(longest_number)


def timeInday(time):
    try:
        h,m = time.split(':')
    except:
        h,m,_ = time.split(':')
    h = int(h)
    m = int(m)   
    minutes = h*60+m
    return minutes/(24*60)


def timeSplit(time):
    time = time[1:-1]
    start, end = time.split(',')
    start = start.strip()
    end = end.strip()
    return (timeInday(start), timeInday(end))


def genDataProcess(trace, map):
    # 轨迹的格式是: [['go to sleep', '(00:00, 09:00)', ['慧忠里小区312号楼', 700764862]],
    # 现在将其处理为：[开始时间点，结束时间点，POIid(字符串), (以米为单位的x,y坐标，通过模拟器的API可以方便地实现)]
    res = []
    for item in trace:
        poiid = item[2]
        poi = map.get_poi(poiid)
        xy = poi['position']
        position = (xy['x'], xy['y'])
        SEtime = timeSplit(item[1])
        res.append([SEtime, poiid, position])
    return res


def readGenTraces(map, folderName):  # 读取多条轨迹
    traces = []
    filePath = 'evaluate/agent/generation/TrajResults/{}'.format(folderName)
    allfiles = os.listdir(filePath)
    success = 0
    for filename in tqdm(allfiles):
        try:
            f = open("evaluate/agent/generation/TrajResults/{}/".format(folderName) + filename, 'r', encoding='utf-8')
            content = f.read()
            oneTrace = json.loads(content)
            
            # 需要的信息：始末时间, POI_id, POI的x,y坐标
            trace = genDataProcess(oneTrace, map)
            traces.append(trace)
            success += 1
        except:
            # print(filename)
            pass
        
    print("read all num: {}".format(success))
    print("actually all num: {}".format(len(allfiles)))
    return traces

def processRealTraces(data, map):
    # 这就是真是数据
    traces = []
    for key, value in data.items():
        trace = []
        for point in value:
            try:
                trace.append([(point[0], point[1]), point[3], map.lnglat2xy(point[4][0], point[4][1])])
            except:
                print(point)
        
        traces.append(trace)
    return traces

