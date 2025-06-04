from openai import OpenAI # >=1.0, test version 1.16.0
import httpx
import os

#### API Key规范使用：避免直接写在代码中，而是作为本地的环境变量
#### 执行方法：在命令中临时执行或者写在.bashrc中
#export SiliconFlow_API_KEY="xx"
#export DeepInfra_API_KEY="xx"
#export OpenAI_API_KEY="xx"


PROXY = "http://127.0.0.1:10190"

#### API平台设置及模型选择
API_KEY_MAPPING = {
    "siliconflow": "SiliconFlow_API_KEY", # 支持模型：https://siliconflow.cn/models
    "DeepInfra": "DeepInfra_API_KEY", # 支持模型：https://deepinfra.com/models
    "OpenAI": "OpenAI_API_KEY" # 支持模型：https://openai.com/api/pricing/
}
API_URL_MAPPING = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "DeepInfra": "https://api.deepinfra.com/v1/openai",
    "OpenAI": "https://api.openai.com/v1"
}
API_TYPE = "siliconflow"
API_KEY = API_KEY_MAPPING[API_TYPE]
API_URL = API_URL_MAPPING[API_TYPE]
model_name = "google/gemma-2-9b-it"

#### 配置client
if API_TYPE == "OpenAI":
    model_name = "gpt-3.5-turbo-0125"
    client = OpenAI(
        base_url=API_URL,
        api_key=API_KEY,
        http_client=httpx.Client(proxies=PROXY)
    )
elif API_TYPE == "siliconflow":
    client = OpenAI(
        base_url=API_URL,
        api_key=API_KEY
    )
elif API_TYPE=="DeepInfra":
    client = OpenAI(
        base_url=API_URL,
        api_key=API_KEY,
        http_client=httpx.Client(proxies=PROXY),
    )


#### 发起请求
dialogs = [{
        "role": "user",
        "content": "请从下面的文字中抽取出地名及其别称，并以JOSON格式输出，比如 [{'地名':'xx', '别称':'xx'}].\n\n 太原，又称龙城，是唐太宗李世民的老家。北京，又称北平，是新中国的首都。石家庄，古称常州，是赵子龙的家乡。"
    }]

completion = client.chat.completions.create(
  model=model_name,
  messages=dialogs,
  max_tokens=100,
  temperature=0
)

print(completion.choices[0].message.content)
