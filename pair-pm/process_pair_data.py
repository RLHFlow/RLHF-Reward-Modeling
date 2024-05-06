from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
import numpy as np

# All the datasets should be pre-processed into standard format.
all_dirs = [
    "RLHFcollection/UltraFeedback-preference-standard",
    "RLHFlow/HH-RLHF-Helpful-standard",
    "RLHFlow/SHP-standard"
]

tokenizer_path = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer_plain = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer_plain.chat_template = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"

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

all_datasets = []
for ds_dir in all_dirs:
    ds = load_dataset(ds_dir, split='train')
    ds_filtered = ds.filter(filter_example, num_proc=32)
    ds_new = ds_filtered.map(process_example,num_proc=32, remove_columns=ds.column_names, )
    all_datasets.append(ds_new)


if len(all_datasets) == 1:
    combined_dataset = all_datasets[0]
else:
    tmp = concatenate_datasets([all_datasets[0], all_datasets[1]])
    for i in range(2, len(all_datasets)):
        tmp = concatenate_datasets([tmp, all_datasets[i]])
    combined_dataset = tmp

combined_dataset = combined_dataset.shuffle(seed=42)


DatasetDict({'train': combined_dataset}).push_to_hub("You own hf dir")