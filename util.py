import pickle
import json
from pathlib import Path
import argparse
import pandas as pd
from pprint import pprint


def load_pkl(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser("")

    # data
    parser.add_argument("--dataset", default='egoschema', type=str)  # 'egoschema', 'nextqa', 'nextgqa', 'intentqa'

    # subset
    parser.add_argument("--data_path", default='path/data/egoschema/lavila_subset.json', type=str) 
    parser.add_argument("--anno_path", default='path/data/egoschema/subset_anno.json', type=str)
    parser.add_argument("--duration_path", default='path/data/egoschema/duration.json', type=str) 

    # # fullset  
    # parser.add_argument("--data_path", default='/data/path/lavila_fullset.json', type=str) 
    # parser.add_argument("--anno_path", default='/data/path/fullset_anno.json', type=str)  
    # parser.add_argument("--duration_path", default='/data/path/duration.json', type=str) 
    parser.add_argument("--fps", default=1.0, type=float) 
    parser.add_argument("--num_examples_to_run", default=-1, type=int)
    ## backup pred
    parser.add_argument("--backup_pred_path", default="", type=str)
    ## fewshot
    parser.add_argument("--fewshot_example_path", default="", type=str) 
    ## nextgqa
    parser.add_argument("--nextgqa_gt_ground_path", default="", type=str)
    parser.add_argument("--nextgqa_pred_qa_path", default="", type=str)

    # output
    parser.add_argument("--output_base_path", default="", type=str)  
    parser.add_argument("--output_filename", required=True, type=str)  

    # tree information
    parser.add_argument("--tree_node_idx", required=True, type=str)  

    # prompting
    parser.add_argument("--model", default="gpt-4-1106-preview", type=str)
    parser.add_argument("--api_key", default="", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--prompt_type", default="qa_standard", type=str)
    parser.add_argument("--task", default="qa", type=str)  # sum, qa, gqa
    ## sum
    parser.add_argument("--num_words_in_sum", default=500, type=int)  

    # other
    parser.add_argument("--disable_eval", action='store_true')
    parser.add_argument("--start_from_scratch", action='store_true')
    parser.add_argument("--save_info", action='store_true')
    parser.add_argument("--save_every", default=10, type=int)

    return parser.parse_args()


def build_fewshot_examples(qa_path, data_path):
    if len(qa_path) == 0 or len(data_path) == 0:
        return None
    qa = load_json(qa_path)
    data = load_json(data_path)  # uid --> str or list 
    examplars = []
    int_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    for i, (uid, examplar) in enumerate(qa.items()):
        description = data[uid]
        if isinstance(description, list):
            description = '. '.join(description)
        examplars.append(f"Examplar {i}.\n Descriptions: {description}.\n Question: {examplar['question']}\n A: {examplar['0']}\n B: {examplar['1']}\n C: {examplar['2']}\n D: {examplar['3']}\n E: {examplar['4']}\n Answer: {int_to_letter[examplar['truth']]}.")
    examplars = '\n\n'.join(examplars)
    return examplars
    
    
    