from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from accelerate import Accelerator
import numpy as np
import torch
from tqdm import tqdm
import argparse
import json
import time
import os
import sys
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_name_or_path", type=str, default='pwork7/llama31_it_prm_2e6_bz32_1epoch_conversation')  # model path
    parser.add_argument("--dataset", type=str, default='RLHFlow/Deepseek-GSM8K-Test')  # data path
    parser.add_argument("--output_dir", type=str, default="math_best_of_n")  # output dir
    parser.add_argument("--num_n", type=int, default=1024)  # number of N for each question
    parser.add_argument("--model_type",type=str,choices=["Mistral","Deepseek"],default='Mistral')
    return parser.parse_args()

def batch_data(data_list, batch_size=8):
    n = batch_size
    batch_data = []
    for i in range(n-1):
        start = i * (len(data_list) // batch_size)
        end = (i+1)* (len(data_list) // batch_size)
        batch_data.append(data_list[start:end])

    last_start = (n-1) * (len(data_list) // batch_size)
    batch_data.append(data_list[last_start:len(data_list)])
    return batch_data

def select_sample(args,sample,model,tokenizer,candidate_tokens,local_rank):
    prompt = sample['prompt']
    scores_list = []
    #text_list = []
    answers = sample['answers'][:args.num_n]
    step_scores = []
    for ans in answers:
        single_step_score = []
        conversation = []
        forward_conv = []
        if args.model_type == "Mistral":
            ans_list = ans.split("ки\n")
        else:
            ans_list = ans.split("\n\n")
        ans_list = [j.strip() for j in ans_list]
        for k in range(len(ans_list)):
            if k == 0:
                text = prompt + " " + ans_list[0]
            else:
                text = ans_list[k]
            conversation.append({"content":text,"role":"user"})
            conversation.append({"content":"<|reserved_special_token_0|>","role":"assistant"})

        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").squeeze(0).to(local_rank)
        indices = torch.where(input_ids == 128002)
        input_ids[indices] = candidate_tokens[0]
        input_ids = input_ids.unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids).logits[:, :, candidate_tokens]  # the +/- is predicted by the positions of the indices
            scores = logits.softmax(dim=-1)[0, :, 0]  # 0 means the prob of + (1 mean -)
            #print(scores)
            mask = indices[0] - 1  # -1 to get the previous token
            single_step_score = scores[mask].detach().to('cpu', dtype=torch.float32).tolist()

        step_scores.append(single_step_score)
        scores_list.append(sum(single_step_score)/len(single_step_score))

    idx = scores_list.index(max(scores_list))
    sample['step_scores'] = step_scores
    return sample['label'][idx] == 1,sample


def worker(args, model, tokenizer, data, local_rank):

    temp_instances = []
    plus_tag_id = tokenizer.encode('+')[-1]
    minus_tag_id = tokenizer.encode('-')[-1]
    candidate_tokens = [plus_tag_id,minus_tag_id]
    for i,sample in enumerate(tqdm(data)):
        sign,new_sample = select_sample(args,sample,model,tokenizer,candidate_tokens,local_rank)
        data[i] = new_sample
        temp_instances.append(sign)

    # Save results
    return temp_instances,data

if __name__ == "__main__":
    args = parse_args()

    accelerator = Accelerator()
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    #print(world_size)
    ds = load_dataset(args.dataset,split="test").select(range(8))
    local_rank = Accelerator().local_process_index
    print("---------------")
    print("begin to load reward model.")
    print("---------------")
    downloaded = False
    while not downloaded:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(args.reward_name_or_path, torch_dtype=torch.bfloat16).to(local_rank).eval()
            downloaded = True
        except Exception as error:
            print("An error occurred:", error)
            print("Failed to load the reward model. Retrying....")
            time.sleep(2)

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    data = []
    data_size = len(ds["prompt"])

    share = int(data_size / world_size) + 1
    ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))
    print(ds)
    for sample in ds:
        data.append(sample)

    selected_data, new_data = worker(args,model,tokenizer,data,local_rank)

    # Send the data to other GPUs
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    all_process_list = [{}] * world_size

    data_to_send = {
        "data": [[selected_data[i]] for i in range(len(selected_data))],
        "new_data": [[new_data[i]] for i in range(len(new_data))]}
    with open(f"{args.output_dir}_{args.num_n}_save_data_{local_rank}.jsonl",'w') as f: # We also save a copy of the step score for each local rank
        for entry in new_data:
            f.write(json.dumps(entry) + "\n")

    import torch.distributed as dist

    dist.all_gather_object(all_process_list, data_to_send)
    gathered_data = []
    gathered_save_data = []

    for i in range(world_size):
        tmp_data = [tmp[0] for tmp in all_process_list[i]["data"]]
        gathered_data.extend(tmp_data)

        tmp_save_data = [tmp[0] for tmp in all_process_list[i]["new_data"]]
        gathered_save_data.extend(tmp_save_data)

    if local_rank == 0:
    #print(len(gathered_data))
        print(f"acc: {sum(gathered_data)/len(gathered_data)}")
        acc = {"accuracy":sum(gathered_data)/len(gathered_data)}

        with open(f"{args.output_dir}_{args.num_n}.json",'w') as f:
            json.dump(acc,f,indent=4,ensure_ascii=False)

        with open(f"{args.output_dir}_{args.num_n}_save_data.jsonl",'w') as f: # We also save a copy of the step score.
            for entry in gathered_save_data:
                f.write(json.dumps(entry) + "\n")
