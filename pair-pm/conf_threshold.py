from typing import List
from datasets import load_dataset,load_from_disk
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm.auto import trange, tqdm
from datasets import concatenate_datasets
from glob import glob


HOME = os.path.expanduser("~")
data_root = "Your data root"
data_name = "Your data name"
folder = f'{data_root}/{data_name}-pred-probs'

# include all subfolders because data might be sharded
ds_subsets = []
for fpath in sorted(glob(f'{folder}/*')):
    print(fpath)
    ds_subsets.append(load_from_disk(fpath))

def filter_example(example):
    
    if len(example['chosen']) % 2 != 0 or len(example['rejected']) % 2 != 0:
        return False
    # must be iteratively 'user' and 'assistant'
    # assert len(example['chosen']) == len(example['rejected'])
    if len(example['chosen']) != len(example['rejected']):
        return False
    n_rounds = len(example['chosen'])
    for i in range(len(example['chosen'])):
        if example['chosen'][i]['role'] != ['user', 'assistant'][i % 2]:
            return False
        if example['rejected'][i]['role'] != ['user', 'assistant'][i % 2]:
            return False
        if len(example['chosen'][i]['content']) == 0 or len(example['rejected'][i]['content']) == 0:
            return False
        if i < n_rounds - 1:
            # chosen and rejected should have the same context
            if example['chosen'][i]['content'] != example['rejected'][i]['content']:
                return False
    return True

def process_example(example):
    chosen = example['chosen']
    rejected = example['rejected']
    # assert chosen[0]['role'] == 'user' and rejected[0]['role'] == 'user'
    # assert len(chosen) % 2 == 0 and len(rejected) % 2 == 0, (len(chosen), len(rejected))
    chosen_position = np.random.randint(2)
    label = ['A', 'B'][chosen_position]
    n_messages = len(chosen)
    assert n_messages == len(rejected)
    context = tokenizer_plain.apply_chat_template(chosen[:-1],tokenize=False)
    
    responses = [chosen[-1]['content'], rejected[-1]['content']]
    response_A = responses[chosen_position]
    response_B = responses[1-chosen_position]
    my_prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
    prompt = my_prompt_template.format(context=context, response_A=response_A, response_B=response_B)
    response = label

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return {"messages": messages, }

ds = concatenate_datasets(ds_subsets)
ds = ds.filter(filter_example)

chosen_probs = np.array(ds['chosen_prob'])
confidence = np.maximum(chosen_probs, 1-chosen_probs,)

# we use gemma-2b-it here as an example
tokenizer_path = "google/gemma-2b-it"
tokenizer_plain = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer_plain.chat_template = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"

ds = ds.map(process_example)

# filter out the examples with low confidence
conf_threshold = 0.8
mask = np.argwhere(confidence > conf_threshold).flatten()
print(f'Confidence threshold: {conf_threshold}')
print(f'Out of the total {len(mask)} data, {len(mask) / len(ds) * 100:.2f}% is selected')
ds_select = ds.select(mask)
ds_select.push_to_hub("You own hf dir")