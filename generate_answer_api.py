import base64
import json
import os
from datasets import load_dataset
from tqdm import tqdm
import requests
from collections import defaultdict
from multiprocessing import Process
from aoss_client import client


SYSTEM_PROMPT = "<|im_start|>system\nReason step by step and place the thought process within the <think></think> tags, and provide the final conclusion at the end.<|im_end|>\n"
# SYSTEM_PROMPT = "Reason step by step and place the thought process within the <think></think> tags, and provide the final conclusion at the end."
# SYSTEM_PROMPT = ""
# 中文 prompt
prompt_choice_direct = """
请根据问题和提供的图片选择正确的答案，并以 <answer>X</answer> 的格式输出答案（X 应该是 A、B、C 或 D）。无需提供任何解释，只需直接输出答案：
1. <answer>X</answer>：最终答案，其中 X 应为 A、B、C 或 D。
输出格式：  
<answer>X</answer> 最终答案，其中 X 应为 A、B、C 或 D。

注意：
2. 不需要解释。
示例：
<answer>A</answer>
"""
prompt_choice_cot = """
请根据问题和提供的图片选择正确答案，并逐步推理输出得出正确答案的思考过程。输出格式应包含两部分：
1. <think>推理过程</think>：逐步的推理过程，可以包括分析、逻辑推理、相关知识的应用等；
2. <answer>X</answer>：最终答案，其中 X 应为 A、B、C 或 D。
注意：
请逐步推理，输出中间推理过程，最后给出答案。

示例： 
<think>题干图形分为上下两组。观察发现，第一组图形均为字母，且按箭头方向，前一个图形最右边的字母出现在下一个图形最左边；第二组图形为数字，规则同上。根据该规律，最左边的数字应为 2，最右边为 4，只有选项 A 符合。</think>  
<answer>A</answer>
"""
prompt_sudoku_direct = '''
请解答以下数独题目。数独的基本要求是：将每个空格填入 1 到 9 之间的任意阿拉伯数字，使得每一行、每一列以及每一个颜色相同的相邻 3×3 小方块中的数字必须是 1 到 9 且不能重复。

输出格式：  
<answer>X</answer> 最终答案，其中 X 为数独的解。无需解释。

注意：
5. <answer>X</answer>：最终答案，其中 X 为数独的答案。
6. 最终答案必须写在 <answer> </answer> 中。
7. 不需要解释。
示例：
<answer>576891432\n243567198\n819234765\n354678219\n687912543\n921345876\n798123654\n465789321\n132456987</answer>
'''
prompt_sudoku_cot = '''
请解答以下数独题目。数独的基本要求是：将每个空格填入 1 到 9 之间的任意阿拉伯数字，使得每一行、每一列以及每一个颜色相同的相邻 3×3 小方块中的数字必须是 1 到 9 且不能重复。

输出格式：  
逐步的思考过程

<answer>X</answer> 最终答案，其中 X 为数独的解。

注意：
1. <answer>X</answer>：最终答案，其中 X 为数独的答案。
2. 最终答案必须写在 <answer> </answer> 中。
示例：
<think>给定输入 [1, 0, 3, 0, 0, 0, 2, 0, 0]，第一行缺少 2，因此变为 [1, 2, 3]。第 1 列有 1 和 2，所以下一个数字是 3；第 2 列已有 2，下一个是 1；剩下的数字是 2。第 3 行中，第 2 列需要 3，第 3 列需要 1。完成后的网格是 [1, 2, 3, 3, 1, 2, 2, 3, 1]。</think>
<answer>123312231</answer>
'''
prompt_raven_direct = '''请解答以下的 Raven 图形推理题。 
输出格式：  
<answer>X</answer> 最终答案，其中 X 表示选项图中子图的位置，按从左到右、从上到下的顺序编号，为一个阿拉伯数字。

注意：
10. <answer>X</answer>：最终答案，其中 X 为选项图中子图的位置，按从左到右、从上到下顺序编号。
11. 最终答案必须写在 <answer> </answer> 中。
12. 不需要解释。
示例：
<answer>3</answer>
'''
prompt_raven_cot = '''请解答以下的 Raven 图形推理题。 
输出格式：  
描述模型图和选项图的详细特征。分析图形中的模式以回答问题。

<answer>X</answer> 最终答案，其中 X 表示选项图中子图的位置，按从左到右、从上到下的顺序编号，为一个阿拉伯数字。

注意：
1. <answer>X</answer>：最终答案，其中 X 为选项图中子图的位置，按从左到右、从上到下顺序编号。
2. 最终答案必须写在 <answer> </answer> 中。
示例：
<think>### 模型图和选项图包含如下基本图形:...\n### 模型图描述：\n第 1 个内容:...\n第 n 个内容:...\n### 选项图描述：\n第 1 个内容:... \n第 n 个内容....\n\n### 答案:3\n</think>
<answer>3</answer>
'''

# 英文 prompt
# prompt_choice_direct = """
# Please select the correct answer based on the question and the provided image, and output the answer in the format <answer>X</answer> (X should be A, B, C, or D). No need to provide any explanations, just output the answer directly:
# 1. <answer>X</answer>: The final answer, where X should be A, B, C, or D.
# Output format:
# <answer>X</answer> The final answer, where X should be A, B, C, or D.

# Note:
# 1. No explanation is needed.

# Example:
# <answer>A</answer>
# """

# prompt_choice_cot = """
# Please select the correct answer based on the question and the provided image, and reason step by step to output the thought process that leads to the correct answer. The output format should include two parts:
# 1. <cot>reasoning process</cot>: The step-by-step reasoning process, which may include analysis, logical reasoning, application of relevant knowledge, etc.
# 2. <answer>X</answer>: The final answer, where X should be A, B, C, or D.

# Note:
# Please reason step by step, output the intermediate reasoning process, and finally provide the answer.

# Example:
# <cot>The question stem graphics are divided into two groups, top and bottom. It is observed that the first group of graphics are all letters, and the rightmost letter in the previous graphic along the arrow direction is on the leftmost side of the adjacent next graphic. The second set of figures are all numbers, and the rightmost number in the previous figure in the direction of the arrow is on the leftmost side of the adjacent figure. According to this rule,? The leftmost digit should be 2 and the rightmost digit should be 4, only option A matches.</cot>
# <answer>A</answer>
# """

# prompt_sudoku_direct='''
# Please solve the following Sudoku puzzle. The basic requirement of Sudoku is to fill each empty space with any Arabic numeral from 1 to 9  such that in each row, each column, and each adjacent 3×3 small square of the same color, the numbers filled in must be from 1 to 9 without any repetition.

# Output format:
# <answer>X</answer>The final answer, where X should be the answer of Sudoku. No explanation is needed.

# Note:
# 1. <answer>X</answer>: The final answer, where X should be the answer of Sudoku.
# 2. the final answer must be in <answer> </answer>
# 3. No explanation is needed.

# Example:
# <answer>576891432\n243567198\n819234765\n354678219\n687912543\n921345876\n798123654\n465789321\n132456987</answer>
# '''

# prompt_sudoku_cot='''
# Please solve the following Sudoku puzzle. The basic requirement of Sudoku is to fill each empty space with any Arabic numeral from 1 to 9  such that in each row, each column, and each adjacent 3×3 small square of the same color, the numbers filled in must be from 1 to 9 without any repetition.

# Output format:
# Step-by-step thinking process

# <answer>X</answer>The final answer, where X should be the answer of Sudoku.

# Note:
# 1. <answer>X</answer>: The final answer, where X should be the answer of Sudoku.
# 2. the final answer must be in <answer> </answer>

# Example:
# <cot>Given the input [1, 0, 3, 0, 0, 0, 2, 0, 0], the first row is missing 2, so it becomes [1, 2, 3]. Column 1 has 1 and 2, so the next value is 3; column 2 has 2, so the next is 1; and the remaining number in row 2 is 2. In row 3, column 2 needs 3 and column 3 needs 1. The completed grid is [1, 2, 3, 3, 1, 2, 2, 3, 1].</cot>
# <answer>123312231</answer>
# '''

# prompt_raven_direct='''Please solve the following raven puzzle.
# Output format:
# <answer>X</answer>The final answer, where X should be the position of the subgraph in the option graph, calculated as the number of subgraphs from left to right, top to bottom, and is an Arabic numeral.

# Note:
# 1. <answer>X</answer>: The final answer, where X should be the position of the subgraph in the option graph, calculated as the number of subgraphs from left to right, top to bottom, and is an Arabic numeral.
# 2. the final answer must be in <answer> </answer>
# 3. No explanation is needed.

# Example:
# <answer>3</answer>

# '''

# prompt_raven_cot='''Please solve the following raven puzzle.
# Output format:
# Describe the fine-grained description model and option diagram. Analyze the patterns in the diagram to answer the question.

# <answer>X</answer>The final answer, where X should be the position of the subgraph in the option graph, calculated as the number of subgraphs from left to right, top to bottom, and is an Arabic numeral.

# Note:
# 1. <answer>X</answer>: The final answer, where X should be the position of the subgraph in the option graph, calculated as the number of subgraphs from left to right, top to bottom, and is an Arabic numeral.
# 2. the final answer must be in <answer> </answer>

# Example:
# <cot>### The model diagram and option diagram include the following basic graphics:... \n### Description of the model diagram:\nContent of the 1st:...\nContent of the n-th:...\n### Description of the option diagram:\nContent of the 1st:... \nContent of the n-th....\n\n### Answer:3\n</cot>
# <answer>3</answer>

# '''



# 配置
# input_json = '/mnt/afs/jingjinhao/project/VisuRiddles/datas/VisuRiddles.json'
input_json = '/mnt/afs/jingjinhao/project/VisuRiddles/datas/VisuRiddles_zh.json'
# result_file = '/mnt/afs/jingjinhao/project/VisuRiddles/result/visuriddles_en_cot.jsonl'
# result_file = '/mnt/afs/jingjinhao/project/VisuRiddles/result/visuriddles_en_dir_v12.jsonl'
# result_file = '/mnt/afs/jingjinhao/project/VisuRiddles/result/visuriddles_zh_cot_v1.jsonl'
# result_file = '/mnt/afs/jingjinhao/project/VisuRiddles/test_result/visuriddles_zh_cot_base_v1_1.jsonl'

result_file = '/mnt/afs/jingjinhao/project/VisuRiddles/result_0827/visuriddles_zh_cot_guide_v1_1.jsonl'
# result_file = '/mnt/afs/jingjinhao/project/VisuRiddles/result/visuriddles_zh_cot_with_guide_v1_2.jsonl'
data_root = '/mnt/afs/jingjinhao/project/Caption/datas'  # 你的图片路径
use_caption = False # 是否使用caption
use_guide = True # 是否使用引导语

# 读取数据集
def load_data(input_json):
    with open(input_json, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        # 如果是以 [ 开头、 ] 结尾，就是一个 JSON 数组
        if text.startswith('[') and text.endswith(']'):
            return json.loads(text)
        # 否则按 JSONL 逐行解析（空行跳过）
        data = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f'第 {i} 行 JSON 解析失败：{e}  内容：{line[:200]}')
        return data

# 将图片转换为base64
def encode_image_to_base64(image_path):
    if 's3' in image_path:
        return base64.b64encode(_aoss_client.get(image_path)).decode("utf-8")
    else:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

# 根据class选择系统prompt
def select_system_prompt(class_name, mode):
    if class_name == 'raven':
        if mode == 'direct':
            return prompt_raven_direct
        elif mode == 'cot':
            return prompt_raven_cot
    elif class_name == 'sudoku':
        if mode == 'direct':
            return prompt_sudoku_direct
        elif mode == 'cot':
            return prompt_sudoku_cot
    else:
        if mode == 'direct':
            return prompt_choice_direct
        elif mode == 'cot':
            return prompt_choice_cot

# API 请求
def api_request(url, messages, class_name, mode):
    query = SYSTEM_PROMPT
    # query += select_system_prompt(class_name, mode) 
    images = []
    for message in messages:
        if message["role"] == "user":
            query += "<|im_start|>user\n"
            if isinstance(message["content"], list):
                query_content = ""
                img_cnt = 0
                for content in message["content"]:
                    if content["type"] == "image_url":
                        query += "<img></img>\n"
                        img_cnt += 1
                        images.append(
                            {"type": "base64", "data": encode_image_to_base64(content["image_url"])}
                        )
                    elif content["type"] == "text":
                        query_content = content["text"]
                    else:
                        raise ValueError("type must be text, image_url")
                query += query_content + "<|im_end|>\n"
            else:
                query += message["content"] + "<|im_end|>\n"
        elif message["role"] == "assistant":
            query += "<|im_start|>assistant\n"
            query += message["content"] + "<|im_end|>\n"
        else:
            raise ValueError("role must be user or assistant")

    query += "<|im_start|>assistant\n"

    play_load = {
        "inputs": query,
        "parameters": {
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "max_new_tokens": 16000,
            "do_sample": True,
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

SUDOKU_JSONL = '/mnt/afs/jingjinhao/project/VisuRiddles/datas/zh_VisuRiddles_caption_sudoku.jsonl'
RAVEN_JSONL  = '/mnt/afs/jingjinhao/project/VisuRiddles/datas/zh_VisuRiddles_caption_raven.jsonl'
DEFAULT_JSONL = '/mnt/afs/jingjinhao/project/VisuRiddles/datas/zh_VisuRiddles_caption_new.jsonl'

# 缓存已加载的索引： class_name -> (exact_dict, basename_dict)
_caption_cache = {}

def load_caption_index(jsonl_path):
    """
    加载 jsonl，建立两个索引：
      exact_dict: 原始字符串 -> caption
      basename_dict: basename(filename) -> caption
    返回 (exact_dict, basename_dict)
    """
    exact = {}
    basename = {}
    if not os.path.exists(jsonl_path):
        return exact, basename
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            caption = j.get('caption') or j.get('Caption') or ''
            # 尝试常见的字段名来取图片标识
            possible_keys = []
            for k in ('image_uri','image_url', 'image', 'img', 'img_url', 'file_name', 'filename', 'image_id', 'url'):
                v = j.get(k)
                if v:
                    possible_keys.append(v)
            # 如果没有明显 key，也尝试从其它字段搜寻字符串（谨慎）
            if not possible_keys:
                # 把整个 json 转为字符串并尝试找常见后缀（可选，不耗时）
                # skip for now
                pass
            for kstr in possible_keys:
                exact[kstr] = caption
                basename[kstr] = caption  # also allow full string as basename key
                bn = os.path.basename(kstr)
                if bn:
                    basename[bn] = caption
    return exact, basename

def get_caption_for_item(item, class_name):
    """
    根据 item 和 class_name 从对应 jsonl 中查找 caption。
    尝试方式（按优先级）：
      2) item['imgs'][0] 原始字符串
      3) basename(item['imgs'][0])
      4) 在 exact_dict / basename_dict 中尝试其它变体
    返回 caption 或 None
    """
    # 选择 jsonl 路径
    if class_name == 'sudoku':
        jsonl_path = SUDOKU_JSONL
    elif class_name == 'raven':
        jsonl_path = RAVEN_JSONL
    else:
        jsonl_path = DEFAULT_JSONL

    # 从缓存读取或加载
    if class_name not in _caption_cache:
        _caption_cache[class_name] = load_caption_index(jsonl_path)
    exact_dict, basename_dict = _caption_cache[class_name]

    # get image identifier from item
    img_field = None
    if 'imgs' in item and item['imgs']:
        img_field = item['imgs'][0]
    else:
        # 可能存在 image_url 字段直接
        img_field = item.get('image_url') or item.get('img') or item.get('image')

    tries = []
    if data_root and img_field:
        tries.append(os.path.join(data_root, img_field))
    if img_field:
        tries.append(img_field)
        tries.append(os.path.basename(img_field))
    # 有时 item 里存的是完整 url
    if item.get('image_url'):
        tries.append(item['image_url'])
        tries.append(os.path.basename(item['image_url']))

    # 去重
    tries = [t for i,t in enumerate(tries) if t and t not in tries[:i]]

    # 尝试匹配
    for t in tries:
        if t in exact_dict:
            return exact_dict[t]
        if t in basename_dict:
            return basename_dict[t]
    # 最后尝试按 basename 匹配索引中的所有键
    bn_try = os.path.basename(img_field) if img_field else ''
    if bn_try and bn_try in basename_dict:
        return basename_dict[bn_try]

    # 未找到
    return None

def get_guide(class_name, mode):
    """
    根据 class_name 返回对应的引导语（仅返回 guide body，不包含前缀 '请根据以下描述回答问题：'）。
    目前为占位模板，后面你可以把占位内容替换为实际的引导语文本。
    """
    if mode == 'cot':
        guides = {
            'sudoku': (
                # 数独专用引导（CoT模式）
                '''
                请解答以下数独题目，并逐步推理出最终答案。数独的基本要求是：每个空格填入 1 到 9 之间的数字，使得每一行、每一列以及每个 3x3 小方块中的数字 1 到 9 不得重复。
                输出格式：
                <think>请按照以下步骤进行推理：首先观察数独中的已填数字，检查每一行、每一列和每个 3x3 小方块中的数字分布情况。然后，逐个空格填入数字，确保符合数独规则。使用排除法逐步填充，最终填满所有空格。</think>
                <answer>X</answer> 最终答案，其中 X 为数独的解。每行数字按顺序排列。
                '''
            ),
            'raven': (
                # Raven 专用引导（CoT模式）
                '''
                请解答以下的 Raven 图形推理题，并逐步推理得出最终答案。
                输出格式：
                <think>首先观察模型图和选项图中的图形，描述每个图形的基本特征，包括形状、位置、旋转、对称、数量等属性。接着比较不同图形之间的变化规律，寻找相似的模式。逐步排除不符合规律的选项，最终确定最符合规律的选项。</think>
                <answer>X</answer> 最终答案，其中 X 表示选项图中子图的位置，按从左到右、从上到下的顺序编号，为一个阿拉伯数字。
                '''
            ),
            'default': (
                '''
                请根据问题和提供的图片选择正确答案，并逐步推理输出得出正确答案的思考过程。输出格式应包含两部分：
                1. `<think>推理过程</think>`：逐步的推理过程，可以包括分析、逻辑推理、相关知识的应用等；
                2. `<answer>X</answer>`：最终答案，其中 X 应为 A、B、C 或 D。

                注意：
                请逐步推理，输出中间推理过程，最后给出答案，以下是可能需要注意的规律。
                1. 位置规律：当图形元素的位置发生变化时，请关注其变化方式，可能涉及平移（确定移动的元素、方向和步数）、旋转（确定旋转方向和角度）或翻转（判断是左右还是上下翻转）。
                2. 样式规律：当图形的组成元素相似时，请关注其运算或排列方式。这可能包括遍历（相同元素的重复出现及分布规律）、加减同异（线条的相加、相减、求同、求异关系）或黑白运算（轮廓相同，但内部颜色不同时的运算规则）。
                3. 属性规律：当图形的整体特征发生变化时，请分析其内在属性。这可能涉及对称性（轴对称或中心对称）、曲直性（线条是全直、全曲还是混合）或开闭性（图形是封闭、开放还是半封闭）。
                4. 数量规律：通过计数寻找模式。需要检查点（如线条交点）、线（直线、曲线、笔画数）、面（封闭区域）或素（独立小图形或部分）的数量变化，由此分析规律。
                5. 空间规律：六面体折纸盒问题：需要识别相对面（同行/同列相隔一个面，或呈“Z”字形两端的面）。相邻面要遵循折叠前后相邻关系不变的原则。三视图：需要明确观察方向（主视图、俯视图、左视图）。截面图：分析常见立体图形的可截面特征。
                如果存在多条规则同时成立，请找出最简的规则组，并以此为决策依据，最终选择最符合的选项；
                最后根据推理步骤得出最终答案，并按要求输出答案格式。

                示例： 
                <think>题干图形分为上下两组。观察发现，第一组图形均为字母，且按箭头方向，前一个图形最右边的字母出现在下一个图形最左边；第二组图形为数字，规则同上。根据该规律，最左边的数字应为 2，最右边为 4，只有选项 A 符合。</think>  
                <answer>A</answer>
                '''
            )
        }
        return guides.get(class_name, guides['default'])
        
    guides = {
        'sudoku': (
            # 数独专用引导
            " 请严格按照数独规则解题：每行、每列、每个 3x3 子格中的数字 1-9 不得重复。"
            " 输出必须只包含答案，格式为 <answer>...</answer>，每行换行表示数独每一行。"
        ),
        'raven': (
            # Raven 专用引导
            " 请首先描述模型图与选项图的关键特征，比较形状、位置、旋转、对称、数量等属性，然后基于规律选择正确的选项编号。"
            " 输出必须只包含答案，格式为 <answer>X</answer>。"
        ),
        'default': (
            # 其他类别的通用引导
            " 请基于图片描述与题干做系统化、结构化的图形规则推理。按下列步骤进行，但无需提供任何解释，只需直接输出答案："
            "1) 首先整体观察图组，识别并列出每幅图的基本元素（形状、线条、封闭区域、点、标记等）、数量与显著属性（比如是否有五角星、黑点、字母或阴影）；"
            "2) 比较相邻或同行/同列图形，寻找规律类型：数量变化（计数封闭区、线条数、元素种类）、属性规律（对称轴、旋转、翻转、朝向、粗细、实心/空心）、位置规律（平移、顺时针/逆时针移动）、样式/叠加（元素叠加/去同存异）、空间重构（立体展开/折叠与面的位置关系）等；"
            "3) 将可能的候选规律用简短的数学或逻辑描述表达出来（例如“每幅图的封闭区域数按 +1 递增”、“第二图是第一图顺时针旋转 90° 并移动一格”）；"
            "4) 用排除法验证每个选项：逐一检验 A/B/C/D 是否满足推断出的规律，优先剔除明显不符的项（如元素数量不对、缺少关键标记、对称性错误或与空间关系矛盾）；"
            "5) 若存在多条并行规则，找出同时成立的最简规则组并以其为决策依据；"
            "6) 最后得出结论并按要求输出答案格式。"
            " 输出必须只包含答案，格式为 <answer>X</answer>（X 为 A/B/C/D 或数字编号）；"
        )
    }
    return guides.get(class_name, guides['default'])

def ask_func(items, url, fw):
    """
    items: list of item dicts (包含 'class', 'imgs', 'question', ...)
    url: 接口地址（传给 api_request）
    fw: 已打开的写文件句柄（用于写入结果 jsonl）
    data_root: 可选，图片的根目录，用于构造与 jsonl 中可能一致的路径
    """
    try:
        for item in tqdm(items):
            class_name = item.get('class', '')
            # 根据需要选择模式（这里默认 direct；如需根据 class 切换，可在此修改）
            # mode = 'direct'  # or
            mode = 'cot'
            combined_text = ''
            caption = ''
            
            
            if use_guide == True:
                combined_text += '/n请根据以下解题方法回答问题：/n' 
                guide_text = get_guide(class_name, mode)
                question = item.get('question', '').strip()
                guide_text = guide_text.strip()
                
                combined_text += guide_text
                
                combined_text += '/n以下是题目正文：/n' 
                combined_text += select_system_prompt(class_name, mode) + '\n'
                
                if question:
                    combined_text += question
                else:
                    combined_text += '问题缺失'
                    print(f"Warning: item {item.get('id', 'unknown')} has no question, using guide only.")

            # 先从对应 jsonl 中查 caption
            if use_caption == True:
                caption = get_caption_for_item(item, class_name)
                if caption is None:
                    if class_name not in ('sudoku', 'raven'):
                        caption = ''
                    else:
                        if 'default' not in _caption_cache:
                            _caption_cache['default'] = load_caption_index(DEFAULT_JSONL)
                        exact_def, basename_def = _caption_cache['default']
                        img_field = item.get('imgs', [None])[0] if item.get('imgs') else item.get('image_url')
                        if img_field:
                            if img_field in exact_def:
                                caption = exact_def[img_field]
                            else:
                                bn = os.path.basename(img_field)
                                caption = basename_def.get(bn, '')
                        else:
                            caption = ''

                # 将 caption 和 question 拼接，加入合适的连接词/标点
                question = item.get('question', '').strip()
                caption_text = caption.strip()
                if question:
                    combined_text = question
                    if combined_text[-1] not in '。.!?！？':
                        combined_text += '。'
                    combined_text += '/n以下是图片的描述：/n'
                    if caption_text:
                        combined_text += caption_text
                    else:
                        combined_text += '（无描述）'
                        print(f"Warning: item {item.get('id', 'unknown')} has no caption, using question only.")
                else:
                    combined_text = caption
                    print(f"Warning: item {item.get('id', 'unknown')} has no question, using caption only.")
            
            if use_caption == False and use_guide == False:
                question = item.get('question', '').strip()
                combined_text += select_system_prompt(class_name, mode) 
                combined_text += question
                
            
            img_path = None
            if item.get('imgs'):
                img_path = os.path.join(data_root, item['imgs'][0]) if data_root else item['imgs'][0]
            else:
                img_path = item.get('image_url') or item.get('img') or None
                
            img_path = os.path.join(data_root, img_path) 
            
            option = item.get('option', '').strip()
            combined_text += "/n以下是备选答案：/n"
            combined_text += option
            combined_text += "/n最终答案必须写在 <answer> </answer> 中。/n"

            message = [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': img_path
                        },
                        {
                            'type': 'text',
                            'text': combined_text
                        }
                    ]
                }
            ]

            # 调用 API 生成答案
            # print(message)
            res = api_request(url, message, class_name, mode)
            pred = f"{res['generated_text'][0].strip()}" if res and 'generated_text' in res else ''

            # 保存结果到 item 并写入文件
            item['pred'] = pred
            # if use_caption:
            #     item['used_caption'] = caption  # 可选：把使用的 caption 也存下来，便于追踪
            fw.write(json.dumps(item, ensure_ascii=False) + '\n')
            fw.flush()

    except Exception as e:
        print('ask_func error:', e)

if __name__ == '__main__':
    # API 地址列表
    ip_list = [
        # "http://10.119.19.240:8000/generate"
        "http://10.119.30.188:8000/generate",
        "http://10.119.30.173:8000/generate",
        "http://10.119.25.124:8000/generate",
        # "http://10.119.20.9:8000/generate"
    ]

    # 读取 VisuRiddles 数据
    items = load_data(input_json)
    # 存储结果的文件
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    fw = open(result_file, 'a', encoding='utf-8')

    # 根据 IP 地址划分进程
    p_num = 28
    cpu_num = len(ip_list) * p_num
    samples_list = [[] for _ in range(cpu_num)]
    for idx, sample in enumerate(items):
        proc_id = idx % cpu_num
        samples_list[proc_id].append(sample)

    # 启动进程进行并行请求
    process = []
    for i in range(cpu_num):
        api_id = i // p_num
        p = Process(target=ask_func, args=(samples_list[i], ip_list[api_id], fw))
        process.append(p)
        p.start()

    for p in process:
        p.join()

    fw.close()