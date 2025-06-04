import json
import jsonlines
import os
import random
import re
import time
import argparse
import traceback
from tqdm import tqdm
from collections import Counter, defaultdict
from openai import OpenAI

DATA_VERSION = "v10"


from serving.llm_api import get_llm_model_client
from config import LLM_MODEL_MAPPING

def api_llama3(prompt, max_retries=3, retry_delay=2):
    # 调用llama3
    model_name = "LLama3-70B"
    client = get_llm_model_client(model_name, infer_server="Siliconflow")
    model_name = LLM_MODEL_MAPPING[model_name]

    attempt = 0
    while attempt < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=prompt,
                max_tokens=1000,
                temperature=0
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")  # Log the error message
            attempt += 1
            if attempt < max_retries:
                print("Retrying...")
                time.sleep(retry_delay)
            else:
                print("All attempts failed. Returning default response.")
                return "Error in generating response"


def parse_answer(assistant_answer):
    # 处理answer格式
    match = re.search(r"Final Answer:\s*(.*?)(?:\.\s*)?$", assistant_answer)
    if match:
        return match.group(1).strip()
    return "No answer provided"


def construct_dialogue_spartqa(base_path, output_jsonl, task_counts):
    input_json = os.path.join(base_path, 'human_train.json')
    
    with open(input_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    total_stories = len(data['data'])

    with jsonlines.open(output_jsonl, mode='w') as writer:
        correct_answers = 0
        total_questions = 0
        with tqdm(total=total_stories, desc='Generating Dialogues', unit='story') as pbar:
            for story_data in data['data']:
                story = story_data['story'][0]

                for question_data in story_data['questions']:

                    q_type = question_data['q_type']
                    question_text = question_data['question']
                    answer_index = question_data['answer']
                    candidates = question_data.get('candidate_answers', [])
                    if answer_index is None or answer_index == "" or (isinstance(answer_index, list) and not answer_index):
                        break
                    if q_type in ['CO', 'FR'] and candidates: 
                        # 处理CO和FR类型的问题
                        if isinstance(answer_index, list) and all(isinstance(i, int) for i in answer_index) and max(answer_index) < len(candidates):
                            answer = ', '.join(candidates[i] for i in answer_index)
                            answer = answer.strip()
                            candidate_text = ", ".join(candidates)
                    
                        prompt = f"{story} {question_text} Please select the correct answer from the provided options ({candidate_text}). If there are multiple correct answers, please select all of them. Provide a step-by-step reasoning followed by your final answer(s), separated by the phrases 'Reasoning:' and 'Final Answer'."

                    elif q_type == 'FB' and candidates:
                        # 处理FB类型的问题
                        candidate_text = ", ".join(candidates)
                        answer = ", ".join(map(str, answer_index)) if isinstance(answer_index, list) else str(answer_index)
                        answer = answer.strip()  
                        prompt = f"{story} {question_text} Please select the correct answer from the provided options ({candidate_text}). If there are multiple correct answers, please select all of them. Provide a step-by-step reasoning followed by your final answer(s), separated by the phrases 'Reasoning:' and 'Final Answer'."

                    
                    else:
                        # 处理NY类型问题
                        answer = ", ".join(map(str, answer_index)) if isinstance(answer_index, list) else str(answer_index)
                        answer = answer.strip()

                        prompt = f"{story} {question_text} Please answer in the simplest form, such as 'Yes' or 'No'(Don't add anything else). Please provide a step-by-step reasoning followed by your final answer, separated by the phrase 'Final Answer:'"

                    single_dialog = [{
                        "role": "user",
                        "content": prompt  
                    }]
                    assistant_answer =api_llama3(single_dialog)  
                    
                    parts = assistant_answer.split("Final Answer:")
                    reasoning = parts[0].strip() if len(parts) > 1 else "No reasoning provided."
                    generated_answer = parse_answer(assistant_answer)

                    # 判断是否Final Answer正确
                    if generated_answer == answer:
                        
                        if candidates:
                            question_text += f" The possible answers are: {candidate_text}."

                        formatted_assistant_answer = f"Answer: {generated_answer}\n{reasoning}"
                        
                        spartqa_session = [
                            {"role": "user", "content": f"{story} {question_text}"},
                            {"role": "assistant", "content": formatted_assistant_answer}
                        ]
                        
                        dialogue_info = {
                            "task": "SPARTQA",
                            "id": f"SPARTQA-{correct_answers}",
                            "diag": spartqa_session
                        }
                        correct_answers += 1
                        writer.write(dialogue_info)

                    total_questions += 1
                    pbar.update(1)

        task_counts['SPARTQA'] = correct_answers
        print(f"SPARTQA total questions: {total_questions}")
        print(f"SPARTQA Accuracy: {correct_answers / total_questions:.2f}")



def process_stepgame(base_path, folder_name, output_file_path, max_samples, start_id, task_counts):
    task_name = "StepGame-" + folder_name.capitalize()
    current_id = start_id
    total_correct = 0
    total_questions = 0

    with tqdm(total=10, desc=f"Processing {task_name}", unit='file') as pbar:
        for i in range(1, 11):
            json_file_path = os.path.join(base_path, folder_name, f"qa{i}_train.json")
            txt_file_path = os.path.join("/data1/liutianhui/CityWorldModel-main/resource/SpatialLM-StepGame/prompts/separate", f"Our_CoT_5shot_clean{i}.txt")

            with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
                prompt = txt_file.read().strip()

            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                current_id, correct_answers, questions_processed = construct_dialogues_stepgame(data, output_file_path, task_name, max_samples, current_id, task_counts, prompt)
                total_correct += correct_answers
                total_questions += questions_processed

            pbar.update(1)  

    if total_questions > 0:
        overall_accuracy = total_correct / total_questions
        print(f"Overall accuracy for {task_name}: {overall_accuracy:.2%}")
    else:
        print("No data processed.")


def construct_dialogues_stepgame(data, output_file, task_name, max_samples, start_id, task_counts, prompt):
    with jsonlines.open(output_file, mode='a') as writer:
        count = start_id
        correct_answers = 0
        questions_processed = 0

        for item in data.values():
            if count - start_id >= max_samples:
                break
            story = " ".join(item['story'])
            question = item['question']
            label = item['label']
            k_hop = item.get('k_hop', None)

            customized_prompt = f"{prompt} {story}\n{question}\nProvide a step-by-step reasoning followed by your final answer(s), separated by the phrases 'Reasoning:' and 'Final Answer'."

            dialogue_entry = {
                "role": "user",
                "content": customized_prompt
            }
            assistant_answer =api_llama3([dialogue_entry])  

            parts = assistant_answer.split("Final Answer:")
            reasoning = parts[0].strip() if len(parts) > 1 else "No reasoning provided."
            generated_answer = parse_answer(assistant_answer)
            formatted_assistant_answer = f"Answer: {generated_answer}\n{reasoning}"

            # 判断是否Final Answer正确
            if generated_answer == label:
                correct_answers += 1
                session_prompt = f" {story} {question}"
                stepgame_session = [
                    {"role": "user", "content": session_prompt},
                    {"role": "assistant", "content": formatted_assistant_answer}
                ]
                
                dialogue_info = {
                    "task": task_name,
                    "id": f"{task_name}-{count}",
                    "diag": stepgame_session
                }
                writer.write(dialogue_info)
                count += 1
            questions_processed += 1

        task_counts[task_name] = count
        return count, correct_answers, questions_processed

        
def construct_dialogue_ReSQ(base_path, output_jsonl, task_counts):
    input_json = os.path.join(base_path, 'train_resq.json')
    with open(input_json, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    total_stories = 266
    correct_answers = 0
    total_questions = 0

    with jsonlines.open(output_jsonl, mode='a') as writer:

        with tqdm(total=total_stories, desc='Generating Dialogues', unit='story') as pbar:
            for item in input_data['data']:
                story_text = " ".join(item['story'])
                if item['questions']:
                    question_data = random.choice(item['questions'])
                    question_text = question_data['question']
                    answer = question_data['answer'][0]

                    prompt = f"{story_text} {question_text} Please answer in the simplest form, such as 'Yes' or 'No'. Please provide a step-by-step reasoning followed by your final answer, separated by the phrase 'Final Answer:'"
                    dialogue = [{"role": "user", "content": prompt}]

                    assistant_answer =api_llama3(dialogue)

                    parts = assistant_answer.split("Final Answer:")
                    reasoning = parts[0].strip() if len(parts) > 1 else "No reasoning provided."
                    generated_answer = parse_answer(assistant_answer)

                    # 判断是否Final Answer正确
                    if generated_answer.lower() == answer.lower():
                        
                        formatted_assistant_answer = f"Answer: {generated_answer}\n{reasoning}"
                        resq_session = [
                            {"role": "user", "content": f"{story_text} {question_text}"},
                            {"role": "assistant", "content": formatted_assistant_answer}
                            
                        ]

                        dialogue_info = {
                            "task": "ReSQ",
                            "id": f"ReSQ-{correct_answers}",
                            "diag": resq_session
                        }
                        correct_answers += 1
                        writer.write(dialogue_info)
                    total_questions += 1

                    
                pbar.update(1)

            task_counts['ReSQ'] = correct_answers
            print(f"ReSQ Accuracy: {correct_answers / total_questions:.2f}")


def score_data(input_path, output_path):
    scores_by_task = defaultdict(Counter)
    
    with jsonlines.open(input_path) as reader, jsonlines.open(output_path, mode='w') as writer:
        progress_bar = tqdm(reader, desc="Processing JSONL")  

        for item in progress_bar:
            try:
                task_type = item['task']
                user_content = item['diag'][0]['content']
                assistant_answer = item['diag'][1]['content']
                parts = assistant_answer.split("Reasoning:")
                answer = parts[0].strip()
                reasoning = parts[1].strip() if len(parts) > 1 else ""

                prompt = f"""
                Here is a spatial reasoning problem and its reasoning process:
                Story and Question: {user_content}
                Answer: {answer}
                Reasoning: {reasoning}

                Please evaluate the reasoning provided above. Score the reasoning from 1 to 10 based on the following criteria:
                - Logical Coherence: Does the reasoning logically follow from the given information, without internal contradictions?
                - Clarity: Is the reasoning clearly articulated, making it easy to understand?
                - Relevance: How well does the reasoning address the specific spatial relationships and details mentioned in the question?
                - Completeness: Does the reasoning cover all necessary aspects to justify the conclusion adequately?
                - Precision in Spatial Descriptions: Are spatial relationships described with accuracy?

                Please provide a numerical score only, without any other formatting.
                """
                # 调用api_llama3 来获取分数
                model_response =api_llama3([{"role": "user", "content": prompt}])
                model_response = model_response.strip().replace('[/ANS]', '').replace('[/INST]', '').replace('[/SOL]', '')
                try:
                    score = float(model_response.strip())  # 尝试转换分数，确保无误
                except ValueError:
                    print("Failed to convert score to float:", model_response)
                    continue

                # 保存分数到数据中并写入文件
                item['score'] = score
                writer.write(item)
                scores_by_task[task_type][score] += 1
                

            except Exception as e:
                print(f"Error processing entry:{traceback.format_exc()}")
                continue
        progress_bar.close()

    # 打印每个分数的计数
    print("Score Frequency Distribution:")
    for task, counter in scores_by_task.items():
        print(f"Score Frequency Distribution for {task}:")
        for score, count in sorted(counter.items()):
            print(f"Score {score}: {count} times")



def filter_data_by_score(input_path, output_path, score_threshold):
    with jsonlines.open(input_path) as reader, jsonlines.open(output_path, mode='w') as writer:
        for item in reader:
            if item.get('score', 0) >= score_threshold:
                # 删除score键，保留其他信息
                del item['score']
                writer.write(item)


def main(args):
    random.seed(42)
    task_counts = {}
    

    # 数据集存储地址
    base_path = "/data1/citygpt/datasets/city_world_model/spatial"
    

    # SPARTQA数据集的对话构造
    construct_dialogue_spartqa(base_path, args.reasoning_file, task_counts)

    # stepgame数据集的对话构造
    # stepgame数据集qa文件采样个数
    qa_samples = 100

    process_stepgame(base_path, "clean", args.reasoning_file, qa_samples, 0, task_counts)
    process_stepgame(base_path, "noise", args.reasoning_file, qa_samples, 0, task_counts)


    # ReSQ数据集的对话构造
    construct_dialogue_ReSQ(base_path, args.reasoning_file, task_counts)

    # 对生成的对话进行评分
    score_data(args.reasoning_file, args.score_file)

    # 筛选过滤数据
    score_threshold = 8.0
    filter_data_by_score(args.score_file, args.output_file, score_threshold)

if __name__ == "__main__":

    # v1.3- STEPGAME 增加了STEPGAME数据集，包含clean和noise两个子文件夹
    # v1.4- SPARTQA 增加了SPARTQA数据集，包含'CO', 'FR', 'YN', 'FB'四种类型的问题
    # v1.5- ReSQ 增加了ReSQ数据集，均为判断题


    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_file", default="simulate/logs/additional_llama3-wudaokou_small-mock-v9-data.jsonl")
    parser.add_argument("--score_file", default="simulate/logs/additional_llama3-wudaokou_small-mock-v9-answer.jsonl")
    parser.add_argument("--output_file", default="simulate/examples/additional-wudaokou_small-mock-v9.jsonl")
    parser.add_argument("--data_version", type=str, default=DATA_VERSION)
    args = parser.parse_args()
    main(args)