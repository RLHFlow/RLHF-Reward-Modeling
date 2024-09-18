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
    "helpsteer-helpfulness",
    "helpsteer-correctness",
    "helpsteer-coherence",
    "helpsteer-complexity",
    "helpsteer-verbosity",
    "ultrafeedback-overall_score",
    "ultrafeedback-instruction_following",
    "ultrafeedback-truthfulness",
    "ultrafeedback-honesty",
    "ultrafeedback-helpfulness",
    "beavertails-is_safe",
    "prometheus-score",
    "argilla-overall_quality",
    "argilla-judge_lm",
    "code-complexity",
    "code-style",
    "code-explanation",
    "code-instruction-following",
    "code-readability",
]

# Initialize the argument parser to handle command-line inputs
parser = ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="sfairXC/FsfairX-LLaMA3-RM-v0.1",
    help="Path to the pre-trained model (HuggingFace path or local folder)",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="RLHFlow/ArmoRM-Multi-Objective-Data-v0.1",
    help="Path to the dataset (HuggingFace path or local folder)",
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
args = parser.parse_args()  # Parse the provided command-line arguments

# Load the specified dataset and prepare it for processing
ds = datasets.load_dataset(args.dataset_path)[
    "train"
]  # Load the training split of the dataset
ds = ds.shuffle(seed=0)  # Shuffle the dataset to ensure randomness
if args.n_shards > 1:
    ds = ds.shard(
        num_shards=args.n_shards, index=args.shard_idx - 1
    )  # Divide dataset into shards if needed

# Load the pre-trained model and tokenizer from the specified path
rm = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16 precision for model weights to save memory
    attn_implementation="flash_attention_2",  # Specify the attention implementation for efficiency
)

device = f"cuda:{args.device}"  # Define the CUDA device string
rm = rm.to(device)  # Move the model to the specified CUDA device
rm_tokenizer = AutoTokenizer.from_pretrained(
    args.model_path
)  # Load the tokenizer associated with the model

# Initialize lists to store embeddings and corresponding labels
embeddings = []
labels = []

# Iterate over each example in the dataset with a progress bar
for example in tqdm(ds, desc="Processing dataset"):
    # Format the conversation messages using the tokenizer's chat template without tokenization
    if args.model_path.endswith("FsfairX-LLaMA3-RM-v0.1"):
        # Follows the demo code: https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1
        conv_formatted = rm_tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        ).replace(rm_tokenizer.bos_token, "")
    else:
        conv_formatted = rm_tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )
    # Tokenize the formatted conversation and move tensors to the specified device
    conv_tokenized = rm_tokenizer(conv_formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        output = rm(**conv_tokenized)  # Forward pass through the model
        # Extract the last hidden state of the last token and move it to CPU
        embeddings.append(output.last_hidden_state[0][-1].cpu())

    # Extract labels for the current example based on predefined attributes
    label = [example[attr] for attr in attributes]
    # Replace None values with NaN for consistent numerical processing
    label = [np.nan if l is None else l for l in label]
    labels.append(label)  # Append the processed labels to the list

# Convert the list of labels to a NumPy array with float32 precision
labels = np.array(labels, dtype=np.float32)
labels = torch.from_numpy(labels)  # Convert the NumPy array to a PyTorch tensor
embeddings = torch.stack(embeddings, dim=0)  # Stack all embeddings into a single tensor

# Define the path to save the embeddings and labels
HOME = os.path.expanduser("~")  # Get the home directory of the current user
model_name = args.model_path.split("/")[
    -1
]  # Extract the model name from the model path
dataset_name = args.dataset_path.split("/")[
    -1
]  # Extract the dataset name from the dataset path
save_path = os.path.join(
    HOME, "data", "ArmoRM", "embeddings", model_name, dataset_name
)  # Construct the save directory path
os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

# Save the embeddings and labels in a safetensors file with shard indexing
save_file(
    {"embeddings": embeddings, "labels": labels},
    f"{save_path}-{args.shard_idx:05d}-of-{args.n_shards:05d}.safetensors",
)

# Print a confirmation message with the path to the saved embeddings
print(
    f"Saved embeddings to {save_path}-{args.shard_idx:05d}-of-{args.n_shards:05d}.safetensors"
)
