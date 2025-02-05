"""
Script to generate embeddings from LLM responses using pre-trained reward models.

This script processes datasets containing prompts and responses, generates embeddings using a specified reward model, and saves the embeddings along with corresponding labels for various attributes like helpfulness, correctness, etc.

Usage:
    python get_embeddings.py [--model_path MODEL] [--dataset_path DATASET] [--split SPLIT] [--n_shards N] [--shard_idx IDX] [--device DEVICE] [--cache_dir PATH] [--save_dir PATH]

Arguments:
    --model_path: Path to pre-trained reward model (default: Skywork/Skywork-Reward-Llama-3.1-8B-v0.2)
    --dataset_path: Path to dataset (default: nvidia/HelpSteer2)
    --split: Dataset split to use (default: train)
    --n_shards: Number of shards to divide dataset into (default: 1)
    --shard_idx: Index of current shard (default: 1)
    --device: CUDA device index (default: 0)
    --cache_dir: Cache directory for model and tokenizer
    --save_dir: Directory to save embeddings
"""

import os
import torch
import datasets
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from safetensors.torch import save_file
from argparse import ArgumentParser

# Set up CUDA optimizations for faster computation
torch.backends.cuda.matmul.allow_tf32 = (
    True  # Enable TensorFloat-32 matrix multiplication on CUDA
)
torch.backends.cudnn.allow_tf32 = (
    True  # Allow TensorFloat-32 in cuDNN for faster convolution operations
)

# Define attributes (reward objectives)
attributes = [
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
]

def convert_to_chat_format(prompt, response):
    if "<extra_id_1>" in prompt:
        """
        Handling HelpSteer2 prompts which may contain multi-turn conversations with the special token <extra_id_1>
        """
        turns = prompt.split("<extra_id_1>")
        conversation = []
        conversation.append({
            "role": "user",
            "content": turns[0].rstrip('\n')
        })
        
        for i in range(1, len(turns)):
            parts = turns[i].split("\n", 1)
            role = parts[0]
            content = parts[1].rstrip('\n')
            conversation.append({
                "role": "assistant" if role == "Assistant" else "user",
                "content": content
            })
    else:
        conversation = [{"role": "user", "content": prompt.rstrip('\n')}]
    conversation.append({"role": "assistant", "content": response.rstrip('\n')})
    return conversation

# Initialize the argument parser to handle command-line inputs
parser = ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    choices=["Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"],
    help="Path to the pre-trained model (HuggingFace path or local folder)",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="nvidia/HelpSteer2",
    choices=["nvidia/HelpSteer2", "allenai/reward-bench"],
    help="Path to the dataset (HuggingFace path or local folder)",
)
parser.add_argument(
    "--split",
    type=str,
    default="train",
    choices=["train", "validation", "filtered"],
    help="Name of the dataset split",
)
parser.add_argument(
    "--n_shards",
    type=int,
    default=1,
    help="Total number of shards to divide the dataset into",
)
parser.add_argument(
    "--shard_idx", type=int, default=1, help="Index of the current shard"
)
parser.add_argument(
    "--device", type=int, default=0, help="CUDA device index to use for computation"
)
parser.add_argument(
    "--cache_dir", type=str, default="/tmp/decision_tree_reward/cache", help="Cache directory for model and tokenizer"
)
parser.add_argument(
    "--save_dir", type=str, default="/tmp/decision_tree_reward/data/", help="Save directory for embeddings and labels"
)
args = parser.parse_args()  # Parse the provided command-line arguments

# Create cache and save directories if they don't exist
os.makedirs(args.cache_dir, exist_ok=True)
os.makedirs(args.save_dir, exist_ok=True)

# Load the specified dataset and prepare it for processing
ds = datasets.load_dataset(args.dataset_path, cache_dir=args.cache_dir)[
    args.split
]  # Load the training split of the dataset


if args.n_shards > 1:
    ds = ds.shard(
        num_shards=args.n_shards, index=args.shard_idx - 1
    )  # Divide dataset into shards if needed

# Load the pre-trained model and tokenizer from the specified path
rm = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16 precision for model weights to save memory
    attn_implementation="flash_attention_2",  # Specify the attention implementation for efficiency
    cache_dir=args.cache_dir,
)

device = f"cuda:{args.device}"  # Define the CUDA device string
rm = rm.to(device)  # Move the model to the specified CUDA device
rm_tokenizer = AutoTokenizer.from_pretrained(
    args.model_path
)  # Load the tokenizer associated with the model

# Initialize lists to store embeddings and corresponding labels
embeddings = []
labels = []

is_pairwise_data = args.dataset_path == "allenai/reward-bench"

if not is_pairwise_data:        
    for example in tqdm(ds, desc="Processing dataset"):
        prompt = example['prompt']
        response = example['response']
        messages = convert_to_chat_format(prompt, response)
        conv_tokenized = rm_tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = rm(conv_tokenized)  # Forward pass through the model
            # Extract the last hidden state of the last token and move it to CPU
            embeddings.append(output.last_hidden_state[0][-1].cpu())

        # Extract labels for the current example based on predefined attributes
        label = [example[attr] for attr in attributes]
        # Replace None values with NaN for consistent numerical processing
        label = [np.nan if l is None else l for l in label]
        labels.append(label)  # Append the processed labels to the list
else:
    for example in tqdm(ds, desc="Processing dataset"):
        prompt = example['prompt']
        pairwise_embedding = []
        for response in [example["rejected"], example["chosen"]]:
            messages = convert_to_chat_format(prompt, response)
            conv_tokenized = rm_tokenizer.apply_chat_template(
                messages, tokenize=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                output = rm(conv_tokenized)
                embedding = output.last_hidden_state[0][-1].cpu()
                pairwise_embedding.append(embedding)
        pairwise_embedding = torch.stack(pairwise_embedding, dim=0)
        embeddings.append(pairwise_embedding)
        labels.append([0,1])
# Convert the list of labels to a NumPy array with float32 precision
labels = np.array(labels, dtype=np.float32)
labels = torch.from_numpy(labels)  # Convert the NumPy array to a PyTorch tensor
embeddings = torch.stack(embeddings, dim=0)  # Stack all embeddings into a single tensor

# Define the path to save the embeddings and labels
model_name = args.model_path.split("/")[
    -1
]  # Extract the model name from the model path
dataset_name = args.dataset_path.split("/")[
    -1
]  # Extract the dataset name from the dataset path
save_path = os.path.join(
    args.save_dir, "embeddings", model_name, dataset_name + "-" + args.split
)  # Construct the save path

os.makedirs(os.path.dirname(save_path), exist_ok=True)
suffix = f"-{args.shard_idx:05d}-of-{args.n_shards:05d}" if args.n_shards > 1 else ""
# Save the embeddings and labels in a safetensors file with shard indexing
save_file(
    {"embeddings": embeddings, "labels": labels},
    f"{save_path}{suffix}.safetensors",
)

# Print a confirmation message with the path to the saved embeddings
print(
    f"Saved embeddings to {save_path}{suffix}.safetensors"
)
