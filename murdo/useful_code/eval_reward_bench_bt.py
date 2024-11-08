from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
import numpy as np
import pandas as pd
tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    data_set_name: Optional[str] = field(
        default='allenai/reward-bench',
        metadata={"help": "the location of the dataset name or path"},
    )
    record_dir: Optional[str] = field(
        default="./bench_mark_eval.txt",
        metadata={"help": "the location of the output file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="sfairXC/FsfairX-LLaMA3-RM-v0.1",
        metadata={"help": "the name of the gold reward model"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

ds_dir = script_args.data_set_name
record_dir = script_args.record_dir 


rm_name = script_args.reward_name_or_path
rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)
device = 0

rm_pipe = pipeline(
    "sentiment-analysis",
    model=rm_name,
    device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    truncation=True
)
pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1,
}



ds = load_dataset(ds_dir, split='filtered', keep_in_memory=True)

df = pd.DataFrame(columns=['id', 'subset', 'correct'])



def change_of_format(prompt, resp):
    message = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": resp}
    ]

    return rm_tokenizer.apply_chat_template(message, tokenize=False).replace(rm_tokenizer.bos_token, "")

def get_reward(test_texts):
    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    return rewards


for i, example in enumerate(tqdm(ds)):

    rewards = get_reward(
                [change_of_format(example['prompt'], example['chosen']), change_of_format(example['prompt'], example['rejected'])]
                )
    
    if rewards[0] == rewards[1]:
        correct = 0.5
    elif rewards[0] > rewards[1]:
        correct = 1.0
    else:
        correct = 0

    row = {'id': example['id'], 'subset': example['subset']}
    row['correct'] = correct
    df = df._append(row, ignore_index=True)

categories = {
    "chat": ["alpacaeval-easy", 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-easy', 'mt-bench-med'],
    "chat-hard": ['mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor', 'llmbar-adver-GPTInst',
                  'llmbar-adver-GPTOut', 'llmbar-adver-manual'],
    "safety": ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 'xstest-should-respond',
               'donotanswer'],
    "reasoning": ['math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust'],
}

df_acc = pd.DataFrame(columns=['category', 'subset', 'accuracy'])
for category, subsets in categories.items():
    for subset in subsets:
        df_subset = df[df['subset'] == subset]
        accs = []
        acc = df_subset['correct'].values.mean()
        accs.append(acc)
        row = {'category': category, 'subset': subset, 'n': len(df_subset), 'accuracy': accs}
        df_acc = pd.concat([df_acc, pd.DataFrame(row)], ignore_index=True)
print(df_acc)

EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 250,
    "xstest-should-respond": 154,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}


def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = round(100 * total_weighted_score / total_examples, 2)
        else:
            section_scores[section] = 0
    return section_scores


all_subsets = df['subset'].unique()
df_final = pd.DataFrame(columns=['attribute', 'Chat', 'Chat Hard', 'Safety', 'Reasoning'])

attribute = 'correct'
metrics = {}
for subset in all_subsets:
    df_subset = df_acc.loc[df_acc['subset'] == subset]
    acc = df_subset['accuracy'].values[0]
    metrics[subset] = acc

# Calculate and print the scores per section
scores_per_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, metrics)
row = {'attribute': attribute, **scores_per_section}
df_final = df_final._append(row, ignore_index=True)
print('model:', script_args.reward_name_or_path)
with open(record_dir, 'a') as f:
    f.write(script_args.reward_name_or_path + "\n")
    for col in ['Chat', 'Chat Hard', 'Safety', 'Reasoning']:
        print(f"{col}: {df_final[col].values[0]}")

        f.write(col + "\t" + str(df_final[col].values[0]) + "\n")
