import json
import os
import re 

def Choice_evalution(item):
    if not isinstance(item['pred'],str):
        return 0
    matches = re.findall(r"<answer>([A-D])</answer>", item['pred'], re.IGNORECASE)
    if not matches:
        print(item)
        return 0

    pred_answer = matches[-1].upper()
    gold_answer = item.get('gold_answer', '').upper()
    
    return 1 if pred_answer == gold_answer else 0

def Sudoku_evalution(item):
    if not isinstance(item['pred'],str):
        return 0
    matches = re.findall(r"<answer>(.*?)</answer>", item.get('pred', ''), re.DOTALL)
    if not matches:
        # print(item.get('pred', ''))
        return 0

    pred_content = matches[-1].lstrip('\n').rstrip('\n')
    pred_content = ''.join(c for c in pred_content if c.isdigit() or c == '\n')
    gold_content = item.get('gold_answer', '')
    
    gold_lines = [line.strip() for line in gold_content.strip().split('\n') if line.strip()]
    if '\n' in pred_content:
        pred_lines = [line.strip() for line in pred_content.strip().split('\n') if line.strip()]
    else:
        pred_lines = pred_content.strip()
        pred_lines = [pred_lines[i:i+len(gold_lines[0])] for i in range(0, len(pred_lines), len(gold_lines[0]))]

    if len(pred_lines) != len(gold_lines):
        print(f'matches:{matches[-1]}')
        print(f'pred_lines:{pred_lines}')
        print(f'gold_lines:{gold_lines}')
        return 0
    
    for pred_line, gold_line in zip(pred_lines, gold_lines):
        if pred_line != gold_line:
            if len(pred_line) != len(gold_line):
                print(pred_lines)
                print(gold_lines)
            return 0

    return 1

def Raven_evalution(item):
    if not isinstance(item['pred'],str):
        return 0
    matches = re.findall(r"<answer>(\d+)</answer>", item.get('pred', ''), re.IGNORECASE)
    if not matches:
        return 0
    pred_answer = matches[-1].strip()
    gold_answer = str(item.get('gold_answer', '')).strip()
    return 1 if pred_answer == gold_answer else 0

def load_data(input_jsonl):
    data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 每行是一个独立的 JSON 对象
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {e}  Line content: {line.strip()}")
    return data

if __name__=='__main__':
    # json_path = r'/mnt/afs/jingjinhao/project/VisuRiddles/result_0827/visuriddles_zh_cot_base_v1_1.jsonl'
    json_path = r'/mnt/afs/jingjinhao/project/VisuRiddles/result_0827/visuriddles_zh_cot_guide_v1_1.jsonl'
    
    # json_path = r'/mnt/afs/jingjinhao/project/VisuRiddles/result/visuriddles_en_dir_v6.jsonl'
    
    # data_list = json.load(open(json_path))
    data_list = load_data(json_path)
    
    print(f"Loaded {len(data_list)} items from the JSONL file.")
    # result_dict = {
    #     'Numerical':[],
    #     'Stylistic':[],
    #     'Attribute':[],
    #     'Positional':[],
    #     'Spatial':[],
    #     'sudoku':[],
    #     'raven':[],
    #     'Other':[],
    #     'All':[]
    # }
    result_dict = {
        '数量规律':[],
        '样式规律':[],
        '属性规律':[],
        '位置规律':[],
        '空间规律':[],
        # 'sudoku':[],
        # 'raven':[],
        '其它':[],
        'All':[]
    }
    for item in data_list:
        acc = 0
        if item['class'] == '立体图':
            item['class'] = '其它'
        
        if item['class'] == '位置关系':
            item['class'] = '位置规律'
        if item['class'] == 'raven':
            acc = Raven_evalution(item)
        elif item['class'] == 'sudoku':
            acc = Sudoku_evalution(item)
        else:
            acc = Choice_evalution(item)
        # print(item['id'])
        if item['class'] != 'sudoku' and item['class'] != 'raven':
            result_dict[item['class']].append(acc)
            result_dict['All'].append(acc)
        
    # result_json_path = '/mnt/afs/jingjinhao/project/VisuRiddles/result/evaluation_results_cot.json'
    # with open(result_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(result_dict, f, ensure_ascii=False, indent=4)
        
    acc_dict = {}
    for category, values in result_dict.items():
        if values:
            avg = sum(values) / len(values)*100
        else:
            avg = 0.0
        
        print(f"{category}: mean = {avg:.2f}")
        acc_dict[category] = round(avg, 2)
    # acc_list = [f"{v:.2f}" for v in acc_list]
    # print('&'.join(acc_list))
    
    output_file = '/mnt/afs/jingjinhao/project/VisuRiddles/result/acc_list.json' 
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(acc_dict, f, ensure_ascii=False, indent=4)
    
        
