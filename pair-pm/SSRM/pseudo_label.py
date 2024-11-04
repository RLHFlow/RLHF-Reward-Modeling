from typing import List
from datasets import load_dataset,load_from_disk
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm.auto import trange, tqdm
import argparse
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument("--shard_idx", type=int, default=0)
parser.add_argument("--n_shards", type=int, default=64)
parser.add_argument("--data_name", type=str)
parser.add_argument("--n_gpu", type=int, default=1)
args = parser.parse_args()

# specify where to save the pseudo-labeled dataset
data_root = 'Your data root'
data_name = args.data_name
dataset_name = 'RLHFlow/UltraFeedback-preference-standard'
max_len = 2048

# Shard data across multiple GPUs for parallel pseudo-labeling
n_shards = args.n_shards
gpu_idx = args.shard_idx % args.n_gpu
shard_idx = args.shard_idx
device = f'cuda:{gpu_idx}'


class SlicPairPMPipeline:

    def __init__(self, model, tokenizer, device):

        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.tokenizer_data_format = AutoTokenizer.from_pretrained(
            "google/gemma-2b-it", use_fast=True
        )
        # self.tokenizer_data_format = AutoTokenizer.from_pretrained(
        #     "meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True
        # )
        x1 = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user"
        x2 = "\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n"
        x3 = " {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"
        my_template = x1 + x2 + x3

        self.tokenizer_data_format.chat_template = my_template

        self.prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
        token_id_A = self.tokenizer.encode("A", add_special_tokens=False)
        token_id_B = self.tokenizer.encode("B", add_special_tokens=False)
        assert len(token_id_A) == 1 and len(token_id_B) == 1
        self.token_id_A = token_id_A[0]
        self.token_id_B = token_id_B[0]
        self.temperature = 1.0
        self.device = device

    def __call__(self, candidates_A: List[str], candidates_B: List[str], **kwargs):
        """
        Input:
            prompts: [prompt1, prompt2, ..., promptn]
            candidates_A: [responseA1, responses A2, ..., responseAn]
            candidates_B: [responseB1, responses B2, ..., responseBn]
        Output:
            probs_choose_A: [P(responseA1 > responseB1 | prompt1), ...., P(responseAn > responseBn | promptn)]
        """

        assert len(candidates_A) == len(candidates_B)
        probs_choose_A = []
        preferences = []
        pred_chosen = []
        pred_rejected = []
        for i in trange(len(candidates_A)):
            chosen = candidates_A[i]
            rejected = candidates_B[i]
            context = self.tokenizer_data_format.apply_chat_template(chosen[:-1], tokenize=False)
            responses = [chosen[-1]["content"], rejected[-1]["content"]]
            probs_chosen = []

            for chosen_position in [0, 1]:
                # we swap order to mitigate position bias
                response_A = responses[chosen_position]
                response_B = responses[1 - chosen_position]
                prompt = self.prompt_template.format(context=context, response_A=response_A, response_B=response_B)
                message = [
                    {"role": "user", "content": prompt},
                ]

                input_ids = self.tokenizer.encode(
                    self.tokenizer.apply_chat_template(message, tokenize=False).replace(self.tokenizer.bos_token, ""),
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.device)

                with torch.no_grad():
                    output = self.model(input_ids)
                logit_A = output.logits[0, -1, self.token_id_A].item()
                logit_B = output.logits[0, -1, self.token_id_B].item()
                # take softmax to get the probability; using numpy
                Z = np.exp(logit_A / self.temperature) + np.exp(logit_B / self.temperature)
                logit_chosen = [logit_A, logit_B][chosen_position]
                prob_chosen = np.exp(logit_chosen / self.temperature) / Z
                probs_chosen.append(prob_chosen)
            prob_chosen_A = np.mean(probs_chosen)
            probs_choose_A.append(prob_chosen_A)
            preference = 0.5 if prob_chosen_A == 0.5 else float(prob_chosen_A > 0.5)
            preferences.append(preference)
        return preferences, probs_choose_A

def get_token_count(example):
    chosen = example["chosen"]
    rejected = example["rejected"]
    context = pipeline.tokenizer_data_format.apply_chat_template(chosen[:-1], tokenize=False)
    responses = [chosen[-1]["content"], rejected[-1]["content"]]

    chosen_position = 0
    # we swap order to mitigate position bias
    response_A = responses[chosen_position]
    response_B = responses[1 - chosen_position]
    prompt = pipeline.prompt_template.format(context=context, response_A=response_A, response_B=response_B)
    message = [
        {"role": "user", "content": prompt},
    ]

    input_ids = pipeline.tokenizer.encode(
        pipeline.tokenizer.apply_chat_template(message, tokenize=False).replace(pipeline.tokenizer.bos_token, ""),
        add_special_tokens=False,
    )
    return {"length": len(input_ids)}

# define the model checkpoint used for pseudo-labeling, we use gemma-2b-it here as an example
model_path = "google/gemma-2b-it"
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2",
                                             device_map=device)
model.eval()
print(f'Loaded model from {model_path} to gpu {device}')
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
pipeline = SlicPairPMPipeline( model, tokenizer, device=device)

# get the shard_idx-th shard of the dataset
ds = load_dataset(dataset_name, split="train")
ds_subset = ds.shard(num_shards=n_shards, index=shard_idx)
print(f'Processing shard {shard_idx} of {n_shards}')
ds_subset = ds_subset.map(get_token_count)
lengths = np.array(ds_subset['length'])
print(f'Data shard of {len(ds_subset)} examples, max length: {lengths.max()}, mean length: {lengths.mean()}')
ds_subset = ds_subset.filter(lambda x: x['length'] <= max_len)
print(f'Filtered examples of token length > {max_len}, remaining {len(ds_subset)} examples')

print("Data preprocessing done. Start pseudo-labeling.")
# get the pseudo-labels
preferences, chosen_probs = pipeline(candidates_A=ds_subset["chosen"], candidates_B=ds_subset["rejected"],)
chosen_probs = np.array(chosen_probs)
# use your own folder name here
folder = f'{data_root}/{data_name}_gemma-2b-pred-probs'
os.makedirs(folder, exist_ok=True)
ds_subset = ds_subset.add_column('chosen_prob',chosen_probs)
ds_subset.save_to_disk(folder+f'/data-shard-{shard_idx+1}-of-{n_shards}')
