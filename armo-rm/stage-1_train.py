import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from safetensors.torch import load_file
from argparse import ArgumentParser


"""
Perform multi-objective linear regression on precomputed embeddings.
This script loads embeddings and labels, splits the data into training and validation sets,
trains Ridge regression models for each attribute across a range of regularization strengths (alphas),
selects the best alpha based on validation loss, and saves the resulting regression weights.
"""

# ---------------------------
# Argument Parsing
# ---------------------------
parser = ArgumentParser(description="Linear Probing on Precomputed Embeddings")
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
    help="Path to the dataset containing multi-objective labels (HuggingFace path or local folder)",
)
parser.add_argument(
    "--embeddings_dir",
    type=str,
    default=None,
    help="Path to the directory containing embedding files. If not provided, defaults to HOME/data/ArmoRM/embeddings/<model_name>/<dataset_name>",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Path to save the regression weights. If not provided, defaults to HOME/data/ArmoRM/regression_weights/",
)
args = parser.parse_args()

# Extract names from paths
args.model_name = args.model_path.split("/")[-1]
args.dataset_name = args.dataset_path.split("/")[-1]

# ---------------------------
# Configuration and Setup
# ---------------------------
# Define the reward attributes as per the method overview in README.md
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

# Set the home directory
HOME = os.path.expanduser("~")

# Define the path to the embeddings based on user input or default location
if args.embeddings_dir:
    embeddings_path = args.embeddings_dir
else:
    embeddings_path = os.path.join(
        HOME, "data", "ArmoRM", "embeddings", args.model_name, args.dataset_name
    )

# Collect all embedding files matching the pattern embeddings_path-*.safetensors
embedding_files = sorted(glob(f"{embeddings_path}-*.safetensors"))

# ---------------------------
# Loading Embeddings and Labels
# ---------------------------
embeddings = []
labels = []
print("Loading embeddings and labels from Safetensors files...")
for file in tqdm(embedding_files, desc="Loading embeddings"):
    # Load the safetensors file
    data = load_file(file)
    embeddings.append(data["embeddings"])  # Append embeddings tensor
    labels.append(data["labels"])  # Append labels tensor

# Concatenate all embeddings and labels into single tensors
embeddings = torch.cat(embeddings, dim=0).float().numpy()
labels = torch.cat(labels, dim=0).float().numpy()

print(f"Total embeddings loaded: {embeddings.shape[0]}")
print(f"Total labels loaded: {labels.shape[0]}")

# ---------------------------
# Splitting Data into Train and Validation Sets
# ---------------------------
print("Splitting data into training and validation sets...")
X_train, X_val, Y_train, Y_val = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# ---------------------------
# Defining Regularization Strengths (Alphas)
# ---------------------------
# Define a range of alpha values for Ridge regression
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
print(f"Using alphas: {alphas}")

# Initialize a DataFrame to store the results of each Ridge regression
df = pd.DataFrame(columns=["attribute", "alpha", "loss"])

# ---------------------------
# Ridge Regression Training
# ---------------------------
print("Training Ridge regression models for each attribute and alpha...")
for attr_idx in tqdm(range(Y_train.shape[1]), desc="Attributes"):
    y_train = Y_train[:, attr_idx]
    # Create a mask to filter out NaN values in training labels
    valid_mask_train = ~np.isnan(y_train)
    y_train_filtered = y_train[valid_mask_train]
    X_train_filtered = X_train[valid_mask_train]

    y_val = Y_val[:, attr_idx]
    # Create a mask to filter out NaN values in validation labels
    valid_mask_val = ~np.isnan(y_val)
    y_val_filtered = y_val[valid_mask_val]
    X_val_filtered = X_val[valid_mask_val]

    # Iterate over each alpha to train Ridge models
    for alpha in tqdm(alphas, desc=f"Alpha for attribute {attr_idx}", leave=False):
        clf = Ridge(alpha=alpha, fit_intercept=False)
        clf.fit(X_train_filtered, y_train_filtered)  # Train the model
        pred = clf.predict(X_val_filtered)  # Predict on validation set
        loss = mean_squared_error(y_val_filtered, pred)  # Calculate MSE loss
        # Append the results to the DataFrame
        df = df._append(
            {"attribute": attr_idx, "alpha": alpha, "loss": loss}, ignore_index=True
        )

# ---------------------------
# Selecting Best Alphas Based on Validation Loss
# ---------------------------
print("Selecting the best alpha for each attribute based on validation loss...")
best_alphas = df.loc[df.groupby("attribute")["loss"].idxmin()]
print("Best alphas selected for each attribute:")
print(best_alphas)

# ---------------------------
# Fitting Final Models with Best Alphas and Extracting Weights
# ---------------------------
print(
    "Fitting final Ridge regression models with the best alphas and extracting weights..."
)
weights = []
for index, row in tqdm(
    best_alphas.iterrows(), total=best_alphas.shape[0], desc="Final Models"
):
    attr_idx = int(row["attribute"])
    best_alpha = row["alpha"]

    # Initialize Ridge model with the best alpha
    clf = Ridge(alpha=best_alpha, fit_intercept=False)

    # Prepare training data
    y_train = Y_train[:, attr_idx]
    valid_mask_train = ~np.isnan(y_train)
    X_train_filtered = X_train[valid_mask_train]
    y_train_filtered = y_train[valid_mask_train]

    # Train the model
    clf.fit(X_train_filtered, y_train_filtered)

    # Append the coefficient (weight) for the current attribute
    weights.append(clf.coef_)

    # Calculate loss on validation set for reporting
    y_val = Y_val[:, attr_idx]
    valid_mask_val = ~np.isnan(y_val)
    X_val_filtered = X_val[valid_mask_val]
    y_val_filtered = y_val[valid_mask_val]
    pred = clf.predict(X_val_filtered)
    loss = mean_squared_error(y_val_filtered, pred)

    print(
        f"Attribute {attr_idx} ({attributes[attr_idx]}): Best alpha = {best_alpha}, Validation Loss = {loss:.4f}"
    )

# Stack all weights into a single NumPy array
weights = np.stack(weights)
print(f"All regression weights shape: {weights.shape}")

# ---------------------------
# Saving the Regression Weights
# ---------------------------
# Define the output directory
if args.output_dir:
    save_dir = args.output_dir
else:
    save_dir = os.path.join(HOME, "data", "ArmoRM", "regression_weights")

os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Define the path to save the weights
save_path = os.path.join(save_dir, f"{args.model_name}_{args.dataset_name}.pt")

# Save the weights as a PyTorch tensor
torch.save({"weight": torch.from_numpy(weights)}, save_path)
print(f"Saved regression weights to {save_path}")
