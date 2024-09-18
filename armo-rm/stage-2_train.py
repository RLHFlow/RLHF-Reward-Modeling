import os
import torch
import numpy as np
from safetensors.torch import load_file
from argparse import ArgumentParser
from tqdm.auto import tqdm
from scipy.stats import spearmanr
import pandas as pd
from glob import glob
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import datasets

# Enable TF32 for improved performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Define the attributes for multi-objective reward modeling
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


class GatingNetwork(nn.Module):
    """
    Gating Network: A simple MLP with softmax output and temperature scaling
    This network learns to combine multiple reward objectives based on the input context
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        temperature: float = 10,
        logit_scale: float = 1.0,
        hidden_dim: int = 1024,
        n_hidden: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
        self.dropout_prob = dropout
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Apply the linear layers with ReLU and dropout
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout_prob > 0:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        # Apply softmax with temperature scaling
        x = F.softmax(x / self.temperature, dim=1)
        return x * self.logit_scale[0]


def find_proper_verbosity_penalties(cluster_V, verbosity_dim=4, corr_threshold=0.028):
    """
    Find appropriate penalties for verbosity to reduce its correlation with other dimensions
    """
    verbosity_penalties = [
        0,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ]
    verbosity_penalties = sorted(verbosity_penalties)
    K = cluster_V.shape[1]
    candidate_dims = set(range(K))
    candidate_dims.remove(verbosity_dim)
    dimwise_verbosity_penalties = np.ones(K)
    dimwise_corr = np.ones(K)
    for verbosity_penalty in verbosity_penalties:
        if len(candidate_dims) == 0:
            break
        V_adjusted = cluster_V - verbosity_penalty * cluster_V[:, [verbosity_dim]]
        corrs = {
            i: spearmanr(V_adjusted[:, i], cluster_V[:, verbosity_dim])[0]
            for i in candidate_dims
        }
        for dim, corr in corrs.items():
            if corr <= corr_threshold:
                candidate_dims.remove(dim)
                dimwise_verbosity_penalties[dim] = verbosity_penalty
                dimwise_corr[dim] = corr
            else:
                dimwise_corr[dim] = np.min([dimwise_corr[dim], corr])
        if len(candidate_dims) == 0:
            break
    return {"penalty": dimwise_verbosity_penalties, "corr": dimwise_corr}


def eval_reward_bench(df_examples, acc_column="correct"):
    """
    Evaluate the model on the RewardBench dataset
    """
    categories = {
        "chat": [
            "alpacaeval-easy",
            "alpacaeval-length",
            "alpacaeval-hard",
            "mt-bench-easy",
            "mt-bench-med",
        ],
        "chat-hard": [
            "mt-bench-hard",
            "llmbar-natural",
            "llmbar-adver-neighbor",
            "llmbar-adver-GPTInst",
            "llmbar-adver-GPTOut",
            "llmbar-adver-manual",
        ],
        "safety": [
            "refusals-dangerous",
            "refusals-offensive",
            "xstest-should-refuse",
            "xstest-should-respond",
            "donotanswer",
        ],
        "reasoning": [
            "math-prm",
            "hep-cpp",
            "hep-go",
            "hep-java",
            "hep-js",
            "hep-python",
            "hep-rust",
        ],
    }

    df_acc = pd.DataFrame(columns=["category", "subset", "accuracy"])
    for category, subsets in categories.items():
        for subset in subsets:
            df_subset = df_examples[df_examples["subset"] == subset]
            acc = df_subset[acc_column].values.mean()
            row = {
                "category": category,
                "subset": subset,
                "n": len(df_subset),
                "accuracy": [acc],
            }
            df_acc = pd.concat([df_acc, pd.DataFrame(row)], ignore_index=True)

    EXAMPLE_COUNTS = {
        "alpacaeval-easy": 100,
        "alpacaeval-length": 95,
        "alpacaeval-hard": 95,
        "mt-bench-easy": 28,
        "mt-bench-med": 40,
        "mt-bench-hard": 37,
        "math-prm": 984,
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

    all_subsets = df_examples["subset"].unique()

    metrics = {}
    for subset in all_subsets:
        df_subset = df_acc.loc[df_acc["subset"] == subset]
        acc = df_subset["accuracy"].values[0]
        metrics[subset] = acc

    scores_per_section = calculate_scores_per_section(
        EXAMPLE_COUNTS, SUBSET_MAPPING, metrics
    )
    score_weights = {"Chat": 1, "Chat Hard": 1, "Safety": 1, "Reasoning": 1}
    scores_per_section["Score"] = round(
        sum([v * score_weights[k] for k, v in scores_per_section.items()])
        / sum(score_weights.values()),
        2,
    )
    return scores_per_section, metrics


def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    """
    Calculate scores for each section of the RewardBench
    """
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = 100 * total_weighted_score / total_examples
        else:
            section_scores[section] = 0
    return section_scores


def load_embeddings(embedding_path_pattern, device):
    """
    Load embeddings from safetensors files
    """
    # Examine if the embedding path pattern is correct
    file_paths = glob(embedding_path_pattern)
    if len(file_paths) == 0:
        raise ValueError(f"Embeddings not found at {embedding_path_pattern}")
    embeddings, prompt_embeddings = [], []
    for embedding_path in file_paths:
        embeddings_data = load_file(embedding_path)
        embeddings.append(embeddings_data["embeddings"].to(device))
        prompt_embeddings.append(embeddings_data["prompt_embeddings"].to(device))
    embeddings = torch.cat(embeddings, dim=0).float()
    prompt_embeddings = torch.cat(prompt_embeddings, dim=0).float()
    return embeddings, prompt_embeddings


# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
parser.add_argument(
    "--multi_objective_dataset",
    type=str,
    default="RLHFlow/ArmoRM-Multi-Objective-Data-v0.1",
)
parser.add_argument(
    "--preference_dataset", type=str, default="RLHFlow/pair_data_v2_80K_wsafety"
)
parser.add_argument(
    "--reference_dataset",
    type=str,
    default=None,
)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--n_steps", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument(
    "--verbosity_dim", type=int, default=4, help="Dimension of the verbosity attribute"
)
parser.add_argument(
    "--corr_threshold",
    type=float,
    default=0.03,
    help="Correlation threshold for verbosity penalty",
)
parser.add_argument("--model_family", type=str, default="llama3", help="Model family")
parser.add_argument(
    "--eval_reward_bench", action="store_true", help="Evaluate on RewardBench"
)
parser.add_argument("--logit_scale", type=float, default=1)
parser.add_argument("--temperature", type=float, default=10)
parser.add_argument("--n_hidden", type=int, default=3)
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--dropout", type=float, default=0.2)
args = parser.parse_args()

# Define default paths
HOME = os.path.expanduser("~")

if args.reference_dataset is None:
    args.reference_dataset = args.preference_dataset
    print(
        f"Using {args.preference_dataset} as the reference dataset for verbosity debiasing."
    )

args.model_name = args.model_path.split("/")[-1]
args.multi_objective_dataset_name = args.multi_objective_dataset.split("/")[-1]
args.preference_dataset_name = args.preference_dataset.split("/")[-1]
args.reference_dataset_name = args.reference_dataset.split("/")[-1]

args.embedding_path = f"{HOME}/data/ArmoRM/embeddings/{args.model_name}/{args.preference_dataset_name}*.safetensors"
args.regression_layer_path = f"{HOME}/data/ArmoRM/regression_weights/{args.model_name}_{args.multi_objective_dataset_name}.pt"
args.reward_bench_embedding_path = (
    f"{HOME}/data/ArmoRM/embeddings/{args.model_name}/reward-bench-filtered.safetensors"
)

device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

# Print the paths for verification
print(f"Embedding path: {args.embedding_path}")
print(f"Regression layer path: {args.regression_layer_path}")
print(f"Reward bench embedding path: {args.reward_bench_embedding_path}")

# Load embeddings
print("Loading embeddings...")
embeddings, prompt_embeddings = load_embeddings(args.embedding_path, device=device)

# Load regression layer
print("Loading regression layer...")
regression_layer = torch.load(args.regression_layer_path, map_location=device)["weight"]

n_attributes, hidden_size = regression_layer.shape

# Load reference dataset embeddings
embedding_path = f"{HOME}/data/ArmoRM/embeddings/{args.model_name}/{args.reference_dataset_name}*.safetensors"
ref_embeddings, ref_prompt_embeddings = load_embeddings(embedding_path, device=device)

# Calculate pairwise rewards and rewards difference
pairwise_rewards = ref_embeddings @ regression_layer.T
rewards = pairwise_rewards.reshape(-1, pairwise_rewards.shape[-1])
rewards_diff = pairwise_rewards[:, 0] - pairwise_rewards[:, 1]

# Find proper verbosity penalties
penalties = find_proper_verbosity_penalties(
    rewards.cpu().numpy().reshape(-1, n_attributes),
    verbosity_dim=args.verbosity_dim,
    corr_threshold=args.corr_threshold,
)
print("Penalties:", penalties)

# Create reward transform matrix
reward_transform_matrix = torch.eye(n_attributes)
reward_transform_matrix[args.verbosity_dim, :] -= torch.from_numpy(penalties["penalty"])
reward_transform_matrix = reward_transform_matrix.to(device)

# Prepare data for training
X = prompt_embeddings  # condition for gating network
Z = embeddings  # multi-objective rewards
R = embeddings @ regression_layer.T @ reward_transform_matrix  # multi-objective rewards
# Split train/val
X_train, X_val, Z_train, Z_val, R_train, R_val = train_test_split(
    X, Z, R, test_size=0.2, random_state=0
)

# Initialize gating network
print("Initializing gating network...")
gating_network = GatingNetwork(
    X_train.shape[-1],
    regression_layer.shape[0],
    n_hidden=args.n_hidden,
    hidden_dim=args.hidden_size,
    logit_scale=args.logit_scale,
    temperature=args.temperature,
    dropout=args.dropout,
).to(device)

# Define loss function and optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    gating_network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps)

# Training loop
print("Starting training...")
for step in tqdm(range(args.n_steps)):
    optimizer.zero_grad()

    # Sample batch
    idx = torch.randint(0, X_train.shape[0], (args.batch_size,))
    X_batch = X_train[idx]
    Z_batch = Z_train[idx]

    # Forward pass
    gating_weights = gating_network(X_batch)
    pred = torch.sum(Z_batch @ regression_layer.T * gating_weights, dim=-1)

    # Compute loss
    loss = loss_fn(pred[:, 0] - pred[:, 1], torch.ones_like(pred[:, 0]))

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Evaluation
print("Evaluating model...")
gating_network.eval()
with torch.no_grad():
    gating_weights_val = gating_network(X_val)
    pred_val = torch.sum(Z_val @ regression_layer.T * gating_weights_val, dim=-1)
    acc_val = ((pred_val[:, 0] - pred_val[:, 1]) > 0).float().mean()
    print(f"Validation accuracy: {acc_val.item():.4f}")

# Save the trained gating network
save_path = f"{HOME}/data/ArmoRM/gating_network_{args.model_name}.pt"
torch.save(gating_network.state_dict(), save_path)
print(f"Saved gating network to {save_path}")

if args.eval_reward_bench:
    # Evaluate on RewardBench
    print("Evaluating on RewardBench...")
    reward_bench_embeddings, reward_bench_prompt_embeddings = load_embeddings(
        args.reward_bench_embedding_path, device=device
    )
    with torch.no_grad():
        gating_weights_rb = gating_network(reward_bench_prompt_embeddings)
        pred_rb = torch.sum(
            reward_bench_embeddings @ regression_layer.T * gating_weights_rb, dim=-1
        )
        correct_rb = (pred_rb[:, 0] > pred_rb[:, 1]).float()

    reward_bench_ds = datasets.load_dataset("allenai/reward-bench", split="filtered")
    df_examples = pd.DataFrame(
        {"subset": reward_bench_ds["subset"], "correct": correct_rb.cpu().numpy()}
    )

    scores_per_section, metrics = eval_reward_bench(df_examples)
    print("RewardBench Scores:")
    print(pd.DataFrame([scores_per_section]))
