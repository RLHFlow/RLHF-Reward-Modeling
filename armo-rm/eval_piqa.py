import json
from tqdm import tqdm
from pipeline import ArmoRMPipeline
from datasets import Dataset, DatasetDict
from datasets import load_dataset, load_from_disk
from argparse import ArgumentParser
import pdb


def validate_answer(sample):
    sample['is_correct'] = sample['rewards'].index(max(sample['rewards'])) == sample['label']
    return sample

def compute_sample_rewards(sample, model):
    prompt = sample['goal']
    options = [x for x in [sample['sol1'], sample['sol2']]]
    scores = []
    for option in options:
        scores.append(model([{"role": "user", "content": prompt}, {"role": "assistant", "content": option}])['score'])
    sample['rewards'] = scores
    return sample

def compute_dataset_rewards(data, model_path):

    rm = ArmoRMPipeline(model_path, trust_remote_code=True)
    kwargs = {
        "model": rm
    }
    data_w_rewards = data.map(compute_sample_rewards, fn_kwargs=kwargs)
    data_w_rewards = data_w_rewards.map(validate_answer)
    accuracy = sum([x['is_correct'] for x in data_w_rewards]) / len(data_w_rewards)
    
    print("Number of samples: ", len(data_w_rewards))
    print(f"Accuracy: {accuracy:.3f}")
    return data_w_rewards


if __name__ == "__main__":

    # Initialize the argument parser to handle command-line inputs
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="RLHFlow/ArmoRM-Llama3-8B-v0.1",
        help="Path to the pre-trained model (HuggingFace path or local folder)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="ybisk/piqa",
        help="Path to the dataset (HuggingFace path or local folder)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../evals/datasets/piqa_w_rewards",
        help="Path to the dataset (HuggingFace path or local folder)",
    )
    args = parser.parse_args()  # Parse the provided command-line arguments

    # Prepare dataset
    data = load_dataset(args.data_path, trust_remote_code=True)
    print(data)

    data_w_rewards = DatasetDict()
    for split in data.keys():
        split_w_rewards = compute_dataset_rewards(data[split], args.model_path)
        data_w_rewards[split] = split_w_rewards

    data_w_rewards.save_to_disk(args.output_path)
