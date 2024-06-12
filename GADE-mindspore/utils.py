from sklearn.metrics import precision_score, recall_score
import json
import os
import pickle
import numpy as np
import mindspore

CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"

def yield_example(ent_list, input_tokens, label_inputs, desc_tokens):
    examples = []
    for e in ent_list:
        example = {
            "entity": e,
            "input_tokens": input_tokens[e],
            "description_token": desc_tokens[e],
            "labels": label_inputs[e]
        }
        examples.append(example)

    return examples



def load_entity_list(path):
    entity_list = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            entity_list.append(line[1])
    return entity_list


def get_kfold_data(ent_list, kfold, i):
    num = len(ent_list)
    fold_size = num // kfold
    train_ent, val_ent, test_ent = None, None, None
    for j in range(kfold):
        idx = slice(j*fold_size, (j+1)*fold_size)
        data_part = ent_list[idx]
        if j == i:
            test_ent = data_part
        elif train_ent is None:
            train_ent = data_part
        else:
            train_ent = train_ent + data_part
    val_ent = train_ent[:fold_size]
    train_ent = train_ent[fold_size:]

    return train_ent, val_ent, test_ent

def generate_data(entity_list, description_files, mention_context_files, tokenizer, max_seq_length):
    with open(description_files, 'r') as f:
        description_json = json.load(f)

    with open(mention_context_files, 'r') as f:
        men_context_json = json.load(f)

    max_num_tokens = max_seq_length - 3
    max_desc_tokens = int(max_num_tokens / 2)
    max_context_tokens = max_num_tokens - max_desc_tokens

    desc_split_tokens = {}
    input_split_tokens = {}
    labels = {}

    for ent in entity_list:
        desc_split_tokens.update({ent: []})
        input_split_tokens.update({ent: []})
        labels.update({ent: []})

    for ent, desc in description_json.items():
        desc_tokens = tokenizer.tokenize(desc)
        if len(desc_tokens) > max_desc_tokens:
            desc_tokens = desc_tokens[:max_desc_tokens]
        desc_split_tokens.update({ent: desc_tokens})

    for ent, men_doc_list in men_context_json.items():
        for name, doc_list in men_doc_list.items():
            for doc_ctx in doc_list:
                start_idx = doc_ctx["start_index"]
                end_idx = doc_ctx["end_index"]
                left_context = doc_ctx["left_context"]
                right_context = doc_ctx["right_context"]
                label = doc_ctx["label"]
                labels[ent].append(label)
                # context contains the mention
                name_token = tokenizer.tokenize(name)
                n_len = len(name_token)
                l_ctx_len = (max_context_tokens - n_len) // 2
                r_ctx_len = max_context_tokens - n_len - l_ctx_len
                left_context_tokens = tokenizer.tokenize(left_context)
                if len(left_context_tokens) > l_ctx_len:
                    left_context_tokens = left_context_tokens[-l_ctx_len:]
                right_context_tokens = tokenizer.tokenize(right_context)
                if len(right_context_tokens) > r_ctx_len:
                    right_context_tokens = right_context_tokens[:r_ctx_len]
                mentions = left_context_tokens + name_token + right_context_tokens
                input_split_tokens[ent].append(mentions)

    return input_split_tokens, labels, desc_split_tokens


def accuracy(pred, label):
    pred = mindspore.ops.argmax(pred, 1).long()
    acc = mindspore.ops.mean((pred == label).float()).asnumpy()
    pred = pred.asnumpy()
    label = label.asnumpy()
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    
    return p, r, acc


def calculate_f1(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pred = (scores > 0.5).astype('int')

    TP = np.sum((pred == 1) * (labels == 1))
    TN = np.sum((pred == 0) * (labels == 0))
    FP = np.sum((pred == 1) * (labels == 0))
    FN = np.sum((pred == 0) * (labels == 1))
    acc = (TP + TN) * 1.0 / (TP + TN + FN + FP)
    if TP == 0:
        p = r = f1 = 0.0
    else:
        p = TP * 1.0 / (TP + FP)
        r = TP * 1.0 / (TP + FN)
        f1 = 2 * p * r / (p + r)

    return p, r, f1, acc


