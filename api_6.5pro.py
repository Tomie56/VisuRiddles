import base64
import json
import requests
from collections import defaultdict
import os
import random
from aoss_client import client
from multiprocessing import Process
from tqdm import tqdm

_aoss_client = client.Client('/mnt/afs/liangjinwei/aoss.conf')


# URLS = {
#     "test1": "http://10.119.21.219:8000/generate",
# }

# # reason
SYSTEM_PROMPT = "<|im_start|>system\nReason step by step and place the thought process within the <think></think> tags, and provide the final conclusion at the end.<|im_end|>\n"
USER_START = "<|im_start|>user\n"
ASSISTANT_START = "<|im_start|>assistant\n"
IM_END = "<|im_end|>\n"
IMG_TAG = "<img></img>\n"
AUDIO_TAG = "<audio></audio>\n"


def encode_image_to_base64(image_path):
    if 's3' in image_path:
        # return get_ceph_img(image_path)
        # return base64.b64encode(io.BytesIO(client.get(image_path))).decode("utf-8")
        return base64.b64encode(_aoss_client.get(image_path)).decode("utf-8")
    else:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def api_request(url, messages):
    query = ""
    if messages[0]["role"] == "system":
        if messages[0]["content"] != "":
            query += "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n"
        messages = messages[1:]
    else:
        query += SYSTEM_PROMPT
    images = []

    for message in messages:
        if message["role"] == "user":
            query += USER_START
            if isinstance(message["content"], list):
                query_content = ""
                img_cnt = 0
                for content in message["content"]:
                    if content["type"] == "image_url":
                        query += IMG_TAG
                        img_cnt += 1
                        images.append(
                            {"type": "base64", "data": encode_image_to_base64(content["image_url"])}
                        )
                    elif content["type"] == "text":
                        query_content = content["text"]
                    else:
                        raise ValueError("type must be text, image_url")
                query += query_content + IM_END
            else:
                query += message["content"] + IM_END
        elif message["role"] == "assistant":
            query += ASSISTANT_START
            query += message["content"] + IM_END
        else:
            raise ValueError("role must be user or assistant")
    query += ASSISTANT_START

    # play_load = {
    #     "inputs": query,
    #     "parameters": {
    #         "temperature": 0.6,
    #         "top_k": 20,
    #         "top_p": 0.95,
    #         "repetition_penalty": 1.1,
    #         "max_new_tokens": 16000,
    #         "do_sample": True,
    #         "add_spaces_between_special_tokens": False,
    #         "skip_special_tokens": True,
    #         "add_special_tokens": False,
    #         "image_patch_max_num":-1,
    #     },
    # }
    play_load = {
        "inputs": query,
        "parameters": {
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "max_new_tokens": 16000,
            "do_sample": True,
            "add_spaces_between_special_tokens": False,
            "skip_special_tokens": True,
            "add_special_tokens": False,
            "image_patch_max_num":-1,
        },
    }
    multimodal_params = {}

    if images:
        multimodal_params["images"] = images
    if multimodal_params:
        play_load["multimodal_params"] = multimodal_params

    headers = {"Content-Type": "application/json"}

    
    # 非流式
    response = requests.post(url, headers=headers, data=json.dumps(play_load))
    response.raise_for_status()
    response = response.json()
    if isinstance(response, list):
        return response[0]["generated_text"]
    return response

def ask_func(items, url):

    try:
        for item in tqdm(items):
            # import pdb;pdb.set_trace()
            res = api_request(url, item[0])
            src_item = item[1]
            # dic = {
            #     'id': item['id'],
            #     'question': item['question'],
            #     'answer': res,
            #     'question_image': item['imgs'][0]['local_path'],
            #     'model': model,
            #     'gt': item['ref_answer']
            # }
            # print(res)
            # dic = {
            #     'image': src_item['image'],
            #     'conversations': [
            #         {
            #             'from': 'human',
            #             'value': '<image>\n' + item[0][0]['content'][1]['text']
            #         },
            #         {
            #             'from': 'gpt',
            #             'value': res['generated_text'][0]
            #         }
            #     ],
            #     'gt': src_item['reward_model']['ground_truth']
            # }
            src_item['result'] = res['generated_text'][0]
            # print(res)
            fw.write(json.dumps(src_item, ensure_ascii=False) + '\n')
            fw.flush()
            
    except Exception as e:
        print(e)

if __name__ == '__main__':

    model = 'rl_landmark_reg_think_250811_3000'
    # input_json = '/mnt/afs/liangjinwei/project/rl/data/ip_data/logo/logo_2kplus_test_QA_zh.jsonl'
    # result_file = f'/mnt/afs/liangjinwei/project/eval/result/logo_test/logo_test_{model}.jsonl'
    input_json = '/mnt/afs/liangjinwei/project/rl/data/ip_data/landmark/landmarks_from_xuyu_test.jsonl'
    result_file = f'/mnt/afs/liangjinwei/project/eval/result/landmarks_from_xuyu_test/landmarks_test_{model}.jsonl'
    # os.makedirs(os.path.dirname(result_file), exists_ok=True)
    # data_root = 's3://mllm_pretrain/Logo_data/'
    data_root = ''
    # prompts = open('/mnt/afs/liangjinwei/project/rl/data/prompt.txt', 'r').read().strip().split('\n')
    exist_dict = defaultdict()
    if os.path.exists(result_file):
        result_file_ = open(result_file, 'r', encoding='utf-8').readlines()
        start_line = len(result_file_)
        print('we start from ', start_line, 'to evaluate')

        for line in result_file_:
            xx = json.loads(line)
            # job_name = xx['answer_name']
            # question = xx['id']
            # job_name = xx['filename']
            #job_name="s3://multi_modal/"+job_name
            exist_dict[xx['image']] = 1
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    fw = open(result_file, 'a')

    messages = []
    
    with open(input_json, 'r') as f:
        for item in f.readlines():
            item = json.loads(item)
            if item['image'] in exist_dict:
                continue
            message = [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': os.path.join(data_root, item['image'])
                        },
                        {
                            'type': 'text',
                            'text': item['prompt'][0]['content']
                        }
                    ]
                }
            ]
            messages.append([message, item])

    ip_list = [
        "http://10.119.20.125:8000/generate",
        "http://10.119.20.94:8000/generate",
        "http://10.119.20.25:8000/generate",
        "http://10.119.20.9:8000/generate"
    ]
    # ask_func(messages, ip_list[0])
    # import pdb;pdb.set_trace()

    print("数据总量为：", len(messages))
    p_num = 30
    cpu_num = len(ip_list) * p_num
    # cpu_num = 3
    samples_list = [[] for _ in range(cpu_num)]
    for idx, sample in enumerate(messages):
        proc_id = idx % cpu_num
        samples_list[proc_id].append(sample)
    
    # import pdb;pdb.set_trace()
    # ask_func(samples_list[0], ip_list[0])
    process = []
    for i in range(cpu_num):
        api_id = i // p_num
        # print(api_id)
        p = Process(target=ask_func, args=(samples_list[i], ip_list[api_id],))
        process.append(p)
        p.start()

    for p in process:
        p.join()
    
    fw.close()

    correct = 0
    with open(result_file, 'r') as f:
        items = f.readlines()
        count = len(items)
        for item in items:
            item = json.loads(item)
            if item['reward_model']['ground_truth'] in item['result']:
                correct += 1

    print(f'总量：', count)
    print(f'准确率：', correct / count)

    

