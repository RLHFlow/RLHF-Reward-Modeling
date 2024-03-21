from transformers import pipeline, AutoTokenizer
import os

from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
import torch.nn as nn
import torch
from typing import Optional
import numpy as np
tqdm.pandas()
from accelerate import Accelerator


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    json_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the dataset name or path"},
    )
    record_dir: Optional[str] = field(
        default="./models/bench_mark_eval.txt",
        metadata={"help": "the location of the output file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="",
        metadata={"help": "the name of the gold reward model"},
    )

names = [
    "Chat",
    "Chat-hard",
    'Safety',
    'Reasoning',
    'anthropic_harmless',
    'anthropic_helpful',
    'summarize',
    'pku_better',
    'pku_safer',
    'shp',
    'anthropic_hhh',
    'mtbench_human',
    'mtbench_gpt4'
]
accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


ds_dir = script_args.json_path
record_dir = script_args.record_dir 


device = accelerator.device

rm_name = script_args.reward_name_or_path
rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)

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


world_size = int(os.getenv("WORLD_SIZE", "1"))

#### TO DO
# Weighted reasoning evaluation


for k in range(13):
    script_args.json_path = str(k+1)

    if int(script_args.json_path) <= 4:
        ds = load_dataset("allenai/reward-bench", split='filtered')
        if script_args.json_path == "1": # chat
            ds = ds.filter(lambda x: x["subset"] in["alpacaeval-easy", 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-easy', 'mt-bench-medium'])
        elif script_args.json_path == "2": # chat hard
            ds = ds.filter(lambda x: x["subset"] in['mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor', 'llmbar-adver-GPTInst', 'llmbar-adver-GPTOut', 'llmbar-adver-manual'])
        elif script_args.json_path == "3": # safety
            ds = ds.filter(lambda x: x["subset"] in['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 'xstest-should-respond', 'do not answer'])
        elif script_args.json_path == "4": # reasoning
            ds = ds.filter(lambda x: x["subset"] in['math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust'])
    else:
        ds = load_dataset("allenai/preference-test-sets", split=names[k])
    print(ds)

    local_rank = Accelerator().local_process_index

    data_size = len(ds['chosen'])
    share = int(data_size / world_size) 
    print(world_size)
    if local_rank < world_size - 1:
        ds = ds.select(np.arange(local_rank * share, (local_rank + 1)*share))
    else:
        ds = ds.select(np.arange(local_rank * share, len(ds)))

    def get_reward(test_texts):
        pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
        rewards = [output[0]["score"] for output in pipe_outputs]
        return rewards





    scores = []
    data = []
    import re

    def change_of_format(prop, resp):
        message = [
            {"role":"user", "content": prop},
            {"role":"assistant", "content": resp}
        ]
        return rm_tokenizer.apply_chat_template(message, tokenize=False).replace(rm_tokenizer.bos_token, "")



    correct_cnt = 0
    all = 0
    for sample in tqdm(ds):
        rewards = get_reward(
            [change_of_format(sample['prompt'], sample['chosen']), change_of_format(sample['prompt'], sample['rejected'])]
            )
        all += 1
        if rewards[0] > rewards[1]:
            correct_cnt += 1
        if rewards[0] == rewards[1]:
            #correct_cnt += 0.5
            pass

    #### Send the data to other GPUs
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    all_process_list =[{}] * world_size

    data_to_send = {
        'correct_cnt': correct_cnt,
        'num_sample': all
    }

    import torch.distributed as dist

    dist.all_gather_object(all_process_list, data_to_send)

    gathered_all_samples = 0
    gathered_all_correct_cnt = 0

    for i in range(world_size):

        gathered_all_samples += all_process_list[i]['num_sample']
        gathered_all_correct_cnt += all_process_list[i]['correct_cnt']




    if local_rank == 0:
        print("Mean accuracy: ", 1.0*gathered_all_correct_cnt/gathered_all_samples)

        fff = 1.0*gathered_all_correct_cnt/gathered_all_samples
        with open(record_dir, 'a') as f:
            f.write(names[k] + "\t" + str(np.mean(fff)) + "\n")
