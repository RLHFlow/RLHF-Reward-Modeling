import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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

def truncate_input_ids(input_ids, score_ids, max_len):
    if input_ids.size(-1) > max_len:
        input_ids = input_ids[:,:max_len]
        score_ids = np.array(score_ids)
        score_ids = score_ids[score_ids <= max_len].tolist()
    return input_ids, score_ids

def select_sample(args,sample,model,tokenizer,candidate_tokens,local_rank, max_len=4096):
    prompt = sample['prompt']
    scores_list = []
    answers = sample['answers'][:args.num_n]
    step_scores = []
    all_status = []
    for ans in answers:
        single_step_score = []
        conversation = []
        if args.model_type == "Mistral":
            ans_list = ans.split("ки\n")
        else:
            ans_list = ans.split("\n\n")
        ans_list = [j.strip() for j in ans_list]
        score_ids = []
        # status: 
        # 0: normal, 
        # 1: truncated due to exceeding max_len, 
        # 2: no complete step after truncation (no score_ids)
        status = 0
        for k in range(len(ans_list)):
            if k == 0:
                text = prompt + " " + ans_list[0]
            else:
                text = ans_list[k]
            conversation.append({"content":text,"role":"user"})
            conversation.append({"content":"+","role":"assistant"})

            # try to concat the token of each part of conversation while recording the position of the reward token
            if k == 0:
                input_ids = tokenizer.apply_chat_template(conversation,return_tensors="pt")
            else:
                input_prompt = tokenizer.apply_chat_template(conversation[-2:], tokenize=False)
                # search the pattern: <xxxx>user<xxxx>
                user_start_pattern = r'<[^<>]+>user<[^<>]+>'
                step_start_idx = re.search(user_start_pattern, input_prompt).start()
                input_prompt = input_prompt[step_start_idx:]
                step_ids = tokenizer(input_prompt,return_tensors="pt",add_special_tokens=False)['input_ids']
                input_ids = torch.cat([input_ids,step_ids],dim=-1)

            # record the predicting reward token position (typically '\n\n')
            score_ids.append(input_ids.size(-1)-3)
        
        # check if the concat input_ids is correct
        ref_input_ids = tokenizer.apply_chat_template(conversation,return_tensors="pt")
        assert torch.all(ref_input_ids == input_ids)

        if input_ids.size(-1) > max_len:
            input_ids, score_ids = truncate_input_ids(input_ids, score_ids, max_len)
            status = 1

        if len(score_ids) == 0:
            single_step_score = [0.0]
            status = 2
        else:
            with torch.no_grad():
                logits = model(input_ids.to(local_rank)).logits[0,:,candidate_tokens]
                logits = logits[score_ids,:]
                scores = logits.softmax(dim=-1)[:,0] # 0 means the prob of + (1 mean -)
                #print(scores)
                single_step_score = scores.detach().to('cpu', dtype=torch.float32).tolist()

        step_scores.append(single_step_score)
        all_status.append(status)
        scores_list.append(sum(single_step_score)/len(single_step_score))

    idx = scores_list.index(max(scores_list))
    sample['step_scores'] = step_scores
    sample['status'] = all_status
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

    accelerator = Accelerator(mixed_precision='bf16')
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    #print(world_size)
    ds = load_dataset(args.dataset,split="test")
    local_rank = accelerator.local_process_index
    print("---------------")
    print("begin to load reward model.")
    print("---------------")
    downloaded = False
    while not downloaded:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(args.reward_name_or_path)
            model = accelerator.prepare(model)
            model.eval()
            downloaded = True
        except Exception as error:
            print("An error occurred:", error)
            print("Failed to load the reward model. Retrying....")
            time.sleep(2)

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    if accelerator.distributed_type == "MULTI_GPU":
        model.module.config.pad_token_id = model.module.config.eos_token_id
    else:
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

    accelerator.wait_for_everyone()
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
