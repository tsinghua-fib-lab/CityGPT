# CityGPT

This repo is for CityGPT: Empowering Urban Spatial Cognition of Large Language Models

## Introduction

In this paper, we propose ***CityGPT***, a systematic framework designed to enhance LLMs' understanding of urban space and improving their ability to solve the related urban tasks by integrating a city-scale ‘world model’ into the model. Firstly, we construct a diverse instruction tuning dataset ***CityInstruction*** for injecting urban knowledge into LLMs and effectively boost their spatial reasoning capabilities. Using a combination of CityInstruction and open source general instruction data, we introduce a two-stage mixed fine-tuning method to train various LLMs (including ChatGLM3-6B, Llama3-8B, and Qwen1.5 series) to enhance their capabilities without compromising their general abilities. To validate the effectiveness of our proposed framework, we develop a comprehensive text-based benchmark ***CityEval*** for evaluating the performance of LLMs across a wide range of urban scenarios and geospatial tasks. Extensive evaluation results demonstrate that smaller LLMs trained with CityInstruction can achieve performance that is competitive with, and in some cases superior to, commercial LLMs when assessed using CityEval. Our work highlights the potential for integrating spatial knowledge into LLMs, thereby expanding their spatial cognition abilities and applicability to the real-world physical environments.

## 🌍 Framework

An overview of CityGPT, including CityInstruction, CityEval and tuning method. We can select any city/region around the world to automatically build new dataset and benchmark for it.
![citygpt](./assets/framework-citygpt.png)

### 🌆 Supported Regions

Currently, the following regions are supported.


| World    | Regions                 | Roads | PoIs |  AoIs  |
| :------- | :---------------------- | :---: | :---: | :----: |
| Asia     | Wudaokou@Beijing        |  148  | 13521 |  1584  |
|          | Wangjing@Beijing        |  470  | 21963 |  6662  |
|          | Yuyuantan@Beijing       |  898  | 50990 | 15324 |
|          | Dahongmen@Beijing       |  358  | 38757 | 10694 |
| Europe   | Paris                   | 4307 | 74303 | 118774 |
| Americas | Lower Manhattan@NewYork |  522  | 11112 | 19541 |

## ⌨️ Codes Structure

- simulate    # codes for constructing training dataset
- train       # codes for fine-tuning models
- evaluate      # evaluation codes
- resource      # basic data of regions
- config.py     # global variables in project

## 🔧 Installation

Install Python dependencies.

```bash
conda create -n citygpt python==3.10
pip install -r requirements.txt
```

## 🤖 LLM Support

For using LLM API, you need to set API Key as follows

```
export OpenAI_API_KEY = ""         # For OpenAI GPT3.5, GPT4, GPT4o
export DeepInfra_API_KEY = ""        # For LLama3, Gemma, Mistral
export SiliconFlow_API_KEY = ""        # For ChatGLM
```

Besides, we use [vllm](https://github.com/vllm-project/vllm) for local LLM deployment.

## 💡 Code Notes

We provide two versions of the code. The branch "CityGPT" is the version used in our paper, while the branch "OSM\_all" is our latest version. The main difference between the two lies in the source of the map data. The former uses External data for the Beijing area and OSM data with only AoI information for other areas, whereas the latter uses OSM data containing full AoI/PoI information for all areas.

## Stage1: Constructing Training Data

Please first set the relevant parameters according to the instructions in `config.py`.

### Existing Dataset of 6 Regions

We provide the CityInstruction dataset for the existing 6 regions respectively. To access the dataset, please refer to [CityGPT-Data-huggingface](https://huggingface.co/datasets/Tianhui-Liu/CityGPT-Data).

### Building a New Region Dataset

If you want to construct a dataset for new regions, please follow the instruction below:

Please first navigate to the `CityWorldModel` directory by using the cd command: `cd CityWorldModel`

#### Basic Data Generation

We provide a script for generating data(including Roads/PoIs/AoIs) in new regions. You need to first define the latitude and longitude range for a city's area in `config.py`, and then run the following command.

```bash
python -m simulate.utils
```

We provide a script that can generate custom-built address for PoI/AoI as well as address directly obtained from OSM.

```bash
python -m simulate.address_system
```

#### CityWalk Construction

First, it is necessary to randomly generate the start and end points of the navigation paths. Please note that the files generated in this step are also required for the  construction of CityQA.

```bash
python -m simulate.train_task
```

In `config.py`, there are 4 parameters related to the CityWalk format. In the training data used for our paper, we utilize the following two versions:

* `EVAL_DATA = False & LANDMARK_DATA = True`
* `EVAL_DATA = False & LANDMARK_DATA = False & DETAIL_INTEREST = True`

In the generation of training data CityReasoning and evaluation benchmark CityEval, the following configuration is also used. Please note that when generating this type of data, make sure to set `DATA_VERSION = "eval"` at the same time.

* `EVAL_DATA = True`

Please first set the desired data parameters in `config.py` according to your needs, and then run the following command.

```bash
python -m simulate.run_simulate_parallel
```

The `run_simulate_parallel.py` script sequentially executes the two codes, `agent.py` and `process.py`, while simulating them in parallel.

#### CityQA Construction

We provide the script for generating the CityQA dataset in parallel. If you want to obtain the dataset more quickly, please run the parallel version of the code directly.

```bash
# Parallel execution
python -m simulate.run_address_data_parallel
```

#### CityReasoning construction

We provide the script to generate the CityReasoning dataset for each task, you can obtain the dataset as the following examples:

```bash
python -m simulate.reasoning_gen
```

Once the above command is executed, you can find the generated training dataset in the `simulate/examples` folder.

## Stage2: Fine-Tuning Models

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to train the model.

### SFT Data Preparation

We provide a script to convert CityInstruction into Alpaca format. You can set the data to be mixed for SFT in `train/scripts/data_process.py`, and then run the script.

```bash
python -m train.scripts.data_process
```

Here, we select influential data for the general benchmarks BBH and CityEval based on [LESS](https://github.com/princeton-nlp/LESS).

### SFT Model

We provide the SFT script. Please adjust the parameters in `examples/train_lora/lora_sft.yaml` as needed before executing the command below.

```bash
./train/run_sft.sh
```

## Stage3: Running CityEval Evaluation

### Evaluation Data Preparation

#### Existing Evaluation Tasks of 6 Regions

We provide the CityEval dataset for the existing 6 regions respectively. To access the dataset, please refer to [CityGPT-Data-huggingface](https://huggingface.co/datasets/Tianhui-Liu/CityGPT-Data).

#### Building CityEval Benchmark for a New Region

If you want to expand the CityEval benchmark for a new area, please run the following command.

```bash
python -m evalaute.task_gen
```

You can find the newly generated tasks in the `evaluate/city_eval/tasks` folder.

### Run the Model Evaluation

#### CI/US/SR Evaluation

We provide a script for evaluation, and the following is an example of how to run it.

```bash
python -m evaluate.city_eval.run_eval \
    --model_name=GPT4o \
    --max_tokens=500 \
    --temperature=0 \
    --city_eval_version=v2 \
    --max_valid=50  \
    --workers=20 \
    --auto_multi \
    --include_answer_prompt_final
```

Here are some new parameters that need to be introduced:

* max\_valid: Refers to the number of examples per task during evaluation.
* include\_answer\_prompt\_final: Adds an "Answer" suffix at the end of the prompt.
* workers: Determines the number of parallel processes.
* auto\_multi: Automatically decides whether to use multi-turn evaluation based on rules.

#### Composite Tasks Evaluation

We provide scripts for three tasks: mobility prediction, trajectory generation, and spatial navigation.

```bash
# For task Mobility Prediction
python -m evaluate.agent.prediction.eval --model=LLama3-8B --mode=gen_answer
# For task Trajectory Generation
python -m evaluate.agent.generation.eval
# For task Spatial Navigation
python -m evaluate.agent.navigation.eval

```

#### General Evaluation

We provide a script for the general evaluation tool OpenCompass in `evaluate/scripts/run.sh`. You can move it to the submodule, modify the content, and run it.

```bash
./evaluate/opencompass-0.2.4/run.sh
```

## 🌟 Citation

If you find this work helpful, please cite our paper.

```latex
@article{Feng2024CityGPTEU,
  title={CityGPT: Empowering Urban Spatial Cognition of Large Language Models},
  author={Jie Feng, Tianhui Liu, Yuwei Du, Siqi Guo, Yuming Lin, and Yong Li},
  journal={ArXiv},
  year={2024},
  volume={abs/2406.13948},
  url={https://api.semanticscholar.org/CorpusID:270619725}
}
```

## 👏 Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.

- https://github.com/princeton-nlp/LESS for selecting influential data
- https://github.com/hiyouga/LLaMA-Factory for fine-tuning model
- https://github.com/THUDM/AgentTuning for multi-choice evaluation
- https://github.com/xlwang233/LLM-Mob for urban mobility prediction

## 📩 Contact

If you have any questions or want to use the code, feel free to contact:
Jie Feng (fengjie@tsinghua.edu.cn)
