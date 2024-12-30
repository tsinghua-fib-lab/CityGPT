import openai
import re
import numpy as np
from thefuzz import process
from datetime import datetime, timedelta

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from evaluate.city_eval.utils import get_chat_completion

def few_shot_exmaples():
    # Q01 = [dict(role="user", content=get_prompts("system") + "\n"+ get_prompts("user").format(data_by_traj_id[traj_id], target_stay_time, my_dict_str, choice_list))]

    info = []

    history = [
        """['2019-10-03 09:00', '天航工业进出口公司'], ['2019-10-01 10:30', '北园食堂'], ['2019-10-01 11:30', '北园食堂'], \
                        ['2019-10-01 15:00', '绿苑小区203号楼'], ['2019-10-01 20:30', '绿苑小区203号楼'], ['2019-10-02 08:30', '北园食堂'], \
                            ['2019-10-02 09:30', '北园食堂'], ['2019-10-02 11:00', '北园食堂'], ['2019-10-02 12:00', '绿苑小区203号楼'], \
                                ['2019-10-02 18:00', '绿苑小区203号楼'], ['2019-10-02 22:00', '绿苑小区203号楼']"""
    ]
    target_time = [
        """2019-10-03 09:00"""
    ]
    candidate_pois = [
        """A.北京航空航天大学武术健美操厅\nB.华盛家园D座\nC.马连洼正黄旗东区\nD.霍营首农龙冠和谐大厦\nE.永泰西里社区北区\nF.天航工业进出口公司\n  """
    ]
    answer = [
        "天航工业进出口公司"
    ]
    # answer = [
    #     """C.御府新景. Based on the provided historical data, it is evident that the user has visited "御府新景" multiple times, indicating a pattern of frequent visits to this location. Given that the target time is 2019-10-08 13:00 and the user was last at "中兴能源" at 10:00 on the same day, it is reasonable to infer that the user may return to "御府新景" after their visit to "中兴能源," as they have done so previously after visiting "中兴能源." Therefore, the prediction for the next Point of Interest (POI) the user is likely to visit at the target time of 2019-10-08 13:00 from the given candidate POIs is C.御府新景.""",
    #     """I.绿苑小区203号楼. Based on the provided historical data, we can observe that the user has a pattern of visiting "北园食堂" multiple times during the day and returning to "绿苑小区203号楼" in the evening. The target time given is 2019-10-02 22:00, which is a time when the user is likely to be at their residence based on the historical pattern. Given this information and the pattern observed, the most likely Point of Interest (POI) the user would visit next at 22:00 on 2019-10-02 would be their residence, which is "绿苑小区203号楼"."""
    # ]
    for i in range(1):
        info.append(dict(role="user", content=get_prompts("user").format(history[i], target_time[i], candidate_pois[i])))
        info.append(dict(role="assistant", content=answer[i]))
    return info

def get_prompts(role):
    system_prompt = """
    Your task is to predict the next Point of Interest (POI) of users based on their historical stays. 
    You will be provided with:
    <history>: containing the user's historical stays in chronological order.
    <target_time>: provides you with the time information for which you need to do next poi inference.
    <candidate POIs>: candidate POIs that the user is predicted to visit.
    Each stay in <history> is represented as [start_time, poi_name].
    Based on these information, you should directly predict the next POI from <candidate POIs>.
    
    Here are several examples:
    """
    user_prompt = """
    Here is the user's historical data and target time to make a prediction:
    <history>: {}
    <target_time>: {}
    <candidate POIs>: {}

    """
    
    if role == "system":
        return system_prompt
    elif role == "user":
        return user_prompt
    else:
        raise NotImplementedError

def get_prompts_new(role):
    system_prompt = """
    Your task is to predict the next Point of Interest (POI) of users based on their historical stays and spatial relationship between POIs.
    You will be provided with:
    <history>: containing the user's historical stays in chronological order,represented as [start_time, poi_name].
    Based on the information provided, you should directly predict the next POI.
    """
    user_prompt = """
    Here is the user's historical data and target time to make a prediction:
    <history>: {}
    <target_time>: {}
    <candidate POIs>: {}
    """
    
    if role == "system":
        return system_prompt
    elif role == "user":
        return user_prompt
    else:
        raise NotImplementedError

def extract_choice(gen, choice_list):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)

    # simply extract the first appearred letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        return choice_list[choice_list.index(process.extractOne(gen, choice_list)[0])]
    
    return res.group(1)

def bleu_score(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    smoothie = SmoothingFunction().method4
    score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    return score


# TODO 从文本中抽取action，与get_available_actions相配合
def process_action(action, choices, limit=0.01, to_print=False):
    if to_print:
        print("preprocess action: ", action)
    match = re.search("ACTION:(.*)", action)
    if match:
        action = match.group(1)
    else:
        return False

    action = action.strip().lower().split("\n")[0]
    if not choices:
        return action
    if action in choices:
        return action
    try:
        bleus = [bleu_score(choice, action) for choice in choices]
        bleus_np = np.array(bleus)
        max_index = np.argmax(bleus_np)
        max_score = bleus[max_index]
        
        # 最大值可能不止一个，取重合度最高的一个
        idx = np.where(bleus_np==max_score)[0]
        if len(idx)>1:
            lens = []
            for i in idx:
                ilen = len(set(choices[i]).intersection(set(action)))
                lens.append(ilen)
            max_idx = np.argmax(lens)
            max_index = idx[max_idx]

        if max_score > limit:
            if to_print:
                print("processed action: ", choices[max_index], " score: ", max_score)
            return choices[max_index]
    except Exception as e:
        print("encounter exception: ", e)
        print("choices: ", choices)
        print("action: ", action)
    
    # 保证不会产生不被允许的action
    return False


def printQA_to_file(Q, A, f):
    print('Question:', Q, file=f)
    print('Answer:', A + '\n', file=f)


def printQA(Q, A):
    print('Question======:', Q)
    print('Answer======:', A + '\n')

def round_to_half_hour(t):
    remainder = t.minute % 30
    if remainder >= 15:
        t += timedelta(minutes=30 - remainder)
    else:
        t -= timedelta(minutes=remainder)
    return t.replace(second=0)

def check_weekend(date_string):
    date_obj = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    # The weekday() function returns 0 for Monday and 6 for Sunday
    if date_obj.weekday() < 5:
        return '工作日'
    else:
        return '休息日'


def askChatGPT(messages, model_name):

    if "gpt" not in model_name:
        model_name, port = model_name.split(":")
    model_name_dict = {
        "gpt-3.5": "gpt-3.5-turbo-0125",
        "gpt-4": "gpt-4-0125-preview",
        "LLama3-70B-AWQ-4bit": "LLama3-70B-AWQ-4bit",
        "chatglm3-6B-v21.4": "chatglm3-6B-v21.4",
        "chatglm3-6B-origin-20240510":"chatglm3-6B-origin-20240510"
    }
    if "gpt" in model_name:
        openai.proxy = "http://127.0.0.1:10190"
        openai.api_key = ''
        
    elif "LLama3-70B-AWQ-4bit" == model_name:
        openai.api_base = "http://xx/v1"
        openai.api_key = ""
    elif "chatglm3-6B-v21.4"==model_name:
        openai.api_base = "http://xx:{}/v1".format(port)
        openai.api_key ="t"
    else:
        openai.api_base = "http://xx:{}/v1".format(port)
        openai.api_key =""

    model_name = model_name_dict[model_name]

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=500,
    )
    # print(response.choices[0].message["content"])

    return response['choices'][0]['message']['content'], response['usage']['total_tokens']

