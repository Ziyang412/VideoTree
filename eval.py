from pathlib import Path
import pandas as pd
from pprint import pprint
from collections import Counter, defaultdict
import argparse
import pdb
from util import *


def eval_qa_egoschema(data):
    num_valids = 0
    num_corrects = 0
    for uid, el in data.items():
        if el['pred'] == -1:
            continue
        num_valids += 1
        if el['truth'] == el['pred']:
            num_corrects += 1 
    stat = {
        'num_total': len(data),
        'num_valids': num_valids,
        'num_corrects': num_corrects,
        'acc': num_corrects / len(data),
    }
    pprint(stat)
    stat['data'] = data
    return stat

def eval_qa_egoschema_from_file(fp):
    data = load_json(fp)
    if 'data' in data:
        data = data['data']
    eval_qa_egoschema(data)

def eval_qa_nextqa(anno_file_path, preds):
    '''
    This function was adapted from https://github.com/doc-doc/NExT-QA/blob/main/eval_mc.py
    '''
    map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When', 'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other', 'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}
    sample_list = pd.read_csv(anno_file_path)
    group = {'CW':[], 'CH':[], 'TN':[], 'TC':[], 'DC':[], 'DL':[], 'DO':[]}
    for id, row in sample_list.iterrows():
        qns_id = str(row['video']) + '_' + str(row['qid'])
        if qns_id not in preds:
            continue
        qtype = str(row['type'])
        #(combine temporal qns of previous and next as 'TN')
        if qtype == 'TP': qtype = 'TN'
        group[qtype].append(qns_id)

    group_acc = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    group_cnt = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    overall_acc = {'C':0, 'T':0, 'D':0}
    overall_cnt = {'C':0, 'T':0, 'D':0}
    all_acc = 0
    all_cnt = 0
    for qtype, qns_ids in group.items():
        cnt = 0
        acc = 0
        for qid in qns_ids:

            cnt += 1
            answer = preds[qid]['truth']
            pred = preds[qid]['pred']

            if answer == pred: 
                acc += 1

        group_cnt[qtype] = cnt
        group_acc[qtype] += acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt

    for qtype, value in overall_acc.items():
        group_acc[qtype] = value
        group_cnt[qtype] = overall_cnt[qtype]

    stat = {}
    for qtype in group_acc:
        print(map_name[qtype], end='\t')
    print('')
    for qtype, acc in group_acc.items():
        if group_cnt[qtype] == 0:
            stat[qtype] = 0
            print('{:.2f}'.format(0), end ='\t')
        else:
            stat[qtype] = acc*100.0/group_cnt[qtype]
            print('{:.2f}'.format(acc*100.0/group_cnt[qtype]), end ='\t')
    stat['Acc'] = all_acc*100.0/all_cnt
    print('')
    print('Acc: {:.2f}'.format(all_acc*100.0/all_cnt))
    stat['data'] = preds
    return stat

def eval_qa_nextqa_from_file(anno_file_path, pred_file_path):
    data = load_json(pred_file_path)
    if 'data' in data:
        data = data['data']
    eval_qa_nextqa(anno_file_path, data)

def eval_sum(data):
    num_words_ls = []
    for example in data.values():
        summarization = example['response']
        num_words = len(summarization.replace('.', ' ').replace(',', ' ').replace('\n', ' ').split(' '))
        num_words_ls.append(num_words)
    num_words_series = pd.Series(num_words_ls)
    stat = {
        'min': float(num_words_series.min()),
        'max': float(num_words_series.max()),
        'mean': float(num_words_series.mean()),
        'std': float(num_words_series.std()),
    }
    stat['data'] = data
    sum_data = {uid: el['response'] for uid, el in data.items()}
    return stat, sum_data

def eval_gqa(gt_ground_path, pred_ground_raw, pred_qa_path=None, subset=None, gs=False):
    '''
    This function is adapted from https://github.com/doc-doc/NExT-GQA/blob/main/code/TempGQA/eval_ground.py
    '''

    def get_tIoU(loc, span):
        if span[0] == span[-1]:
            if loc[0] <= span[0] and span[0] <= loc[1]:
                return 0, 1
            else:
                return 0, 0
        
        span_u =  (min(loc[0], span[0]), max(loc[-1], span[-1]))
        span_i = (max(loc[0], span[0]), min(loc[-1], span[-1]))
        dis_i = (span_i[1] - span_i[0])
        if span_u[1] > span_u[0]:
            IoU = dis_i / (span_u[1] - span_u[0]) 
        else: 
            IoU = 0.0
        if span[-1] > span[0]:
            IoP = dis_i / (span[-1] - span[0]) 
        else:
            IoP = 0.0

        return IoU, IoP

    def get_tIoU_multi(loc, spans):
        overlap = 0
        loc_start, loc_end = loc[0], loc[1]
        for [span_start, span_end] in spans:
            if span_start > span_end:
                span_start, span_end = span_end, span_start
            # if span_end < loc_start or span_start > loc_end:
            #     continue
            overlap_start, overlap_end = max(span_start, loc_start), min(span_end, loc_end)
            if overlap_start >= overlap_end:
                continue
            overlap += abs(overlap_end-overlap_start)
        gt = loc[1] - loc[0]
        pred = sum(map(lambda x: abs(x[1] - x[0]), spans))

        IoU = overlap / (gt + pred - overlap)
        IoP = overlap / pred
        return IoU, IoP

    gt_ground = load_json(gt_ground_path)
    pred_qa = load_json(pred_qa_path) if pred_qa_path else None
    if pred_qa:
        if 'data' in pred_qa:
            pred_qa = pred_qa['data']
    pred_ground = {quid: el['pred'] for quid, el in pred_ground_raw.items()}
    mIoU, mIoP = 0, 0
    cnt, cqt = 0, 0
    crt3, crt5 = 0, 0
    crtp3, crtp5 = 0, 0
    for vid, anno in gt_ground.items():
        for qid, locs in anno['location'].items():
            if not (f'{vid}_{qid}' in pred_ground):
                # print(vid, qid)
                continue
            if subset != None:
                # Non-Blind and Non-Sig QA subset
                if not (f'{vid}_{qid}' in subset):
                    continue
            max_tIoU, max_tIoP = 0, 0
            for loc in locs:
                span = pred_ground[f'{vid}_{qid}']
                # we need to multiply video duration if Gaussian
                if gs: span = np.round(np.asarray(span)*anno['duration'], 1)
                span = span[0]
                tIoU, tIoP = get_tIoU(loc, span)
                # span = [span[0]]
                # tIoU, tIoP = get_tIoU_multi(loc, span)
                if tIoU > max_tIoU:
                    max_tIoU = tIoU
                if tIoP > max_tIoP:
                    max_tIoP = tIoP
            if max_tIoP >= 0.3:
                crtp3 += 1
                if  max_tIoP >= 0.5:
                    crtp5 += 1
                    kid = f'{vid}_{qid}'
                    
                    if pred_qa:
                        if pred_qa[kid]['truth'] == pred_qa[kid]['pred']:
                            cqt+= 1
                            # print(kid)

            if max_tIoU >= 0.3:
                crt3 += 1
                if max_tIoU >= 0.5:
                    crt5 += 1
                    # if pred_qa:
                    #     if pred_qa[kid]['answer'] == pred_qa[kid]['prediction']:
                    #         print(kid)

            cnt += 1
            mIoU += max_tIoU
            mIoP += max_tIoP
    
    mIoU = mIoU /cnt * 100
    mIoP = mIoP/cnt * 100
    print('Acc&GQA mIoP TIoP@0.3 TIoP@0.5 mIoU TIoU@0.3 TIoU@0.5 ')
    print('{:.1f} \t {:.1f}\t {:.1f}\t {:.1f} \t {:.1f} \t {:.1f} \t {:.1f}'.format(cqt*1.0/cnt*100, mIoP, crtp3*1.0/cnt*100, crtp5*1.0/cnt*100, mIoU, crt3*1.0/cnt*100, crt5*1.0/cnt*100))

    stat = {
        'Acc_GQA': cqt*1.0/cnt*100,
        'mIoP': mIoP,
        'TIoP_0.3': crtp3*1.0/cnt*100,
        'TIoP_0.5': crtp5*1.0/cnt*100,
        'mIoU': mIoU,
        'TIoU_0.3': crt3*1.0/cnt*100,
        'TIoU_0.5': crt5*1.0/cnt*100
    }
    stat['data'] = pred_ground_raw

    return stat


def eval_gqa_from_file(gt_ground_path, pred_ground_path, pred_qa_path=None):
    pred_ground = load_json(pred_ground_path)
    if 'data' in pred_ground:
        pred_ground = pred_ground['data']
    eval_gqa(gt_ground_path, pred_ground, pred_qa_path=pred_qa_path)


def eval_egoschema_cats(data_path, cats_path):
    data = load_json(data_path)
    if 'data' in data:
        data = data['data']
    cats = load_json(cats_path)
    cats = {el[1]: el[-1] for el in cats}  # uid --> [cat0, cat1, ...]

    def eval(preds):
        num_corrects = defaultdict(int)  # q_type --> int
        num_total = defaultdict(int)  # q_type --> int
        for uid, info in preds.items():
            q_type_list = info['type']
            pred = info['pred']
            truth = info['truth']
            for q_type in q_type_list:
                if pred == -1:
                    num_corrects[q_type] += 0.2
                else:
                    num_corrects[q_type] += (pred==truth)
                num_total[q_type] += 1
        accs = {k: num_corrects[k] / num_total[k] for k in num_corrects}
        acc_all = sum(list(num_corrects.values())) / sum(list(num_total.values()))
        return accs, acc_all

    for k, v in cats.items():
        for el in v:
            if el not in [1, 2, 3, 4, 5]:
                print('question category not found: ', k)

    # category stat
    id_to_name = {
        1: 'Purpose/Goal Identification',
        2: 'Character Interaction',
        3: 'Tools and Materials Usage',
        4: 'Key Action/Moment Detection',
        5: 'Action Sequence Analysis'
    }
    arr = sum(list(cats.values()), [])
    stat = Counter(arr).most_common()
    print('Category Statistics:')
    for q_type, count in stat:
        print(f"{id_to_name[q_type]}: {count / len(cats) * 100:.1f}")
    print()

    # eval
    preds = {uid: {'pred': uid_info['pred'], 'truth': uid_info['truth'], 'type': cats[uid]} for uid, uid_info in data.items() if uid in cats}
    accs, acc_all = eval(preds)
    accs = sorted(list(accs.items()))

    print('Evaluation:')
    for k, v in accs:
        print(f"{id_to_name[k]}: {v*100:.1f}")
    print()
    print(f"all: {acc_all*100:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function",
        required=True,
        type=str,
    )
    args, unknown = parser.parse_known_args()
    function_arg_names = unknown[0::2]
    function_arg_values = unknown[1::2]
    function_args = {function_arg_names[i]: function_arg_values[i] for i in range(len(function_arg_names))}
    print()
    globals()[args.function](**function_args)


'''
python eval.py \
--function eval_egoschema_cats \
data_path output/egoschema/standard_qa.json \
cats_path data/egoschema/categories.json

python eval.py \
--function eval_qa_nextqa_from_file \
anno_file_path data/nextgqa/test.csv \
pred_file_path output/nextgqa/gpt4_llava.json

python eval.py \
--function eval_gqa_from_file \
gt_ground_path data/nextgqa/gsub_test.json \
pred_qa_path output/nextgqa/gpt4_llava.json \
pred_ground_path output/nextgqa/gpt4_llava_grounding.json
'''