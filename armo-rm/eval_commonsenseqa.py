import json
from tqdm import tqdm
from pipeline import ArmoRMPipeline
from datasets import load_dataset, load_from_disk
from argparse import ArgumentParser


def validate_answer(scores, answer):
    labels = ['A', 'B', 'C', 'D', 'E']
    assert len(scores) == len(labels)
    return scores.index(max(scores)) == labels.index(answer)

def compute_sample_rewards(model, sample):

    prompt = sample['question']['stem']
    options = [x['text'] for x in sample['question']['choices']]

    scores = []
    for option in options:
        scores.append(model([{"role": "user", "content": prompt}, {"role": "assistant", "content": option}])['score'])
    sample['rewards'] = scores
    return sample

def compute_dataset_rewards(data_path, model_path):

    data = [json.loads(x) for x in open(data_path)]
    rm = ArmoRMPipeline(model_path, trust_remote_code=True)

    data_w_rewards = []
    correct_cnt = 0
    for sample in tqdm(data):
        sample_w_rewards = compute_sample_rewards(rm, sample)
        #print(sample_w_rewards["rewards"], sample_w_rewards["answerKey"])
        correct_cnt += validate_answer(sample_w_rewards["rewards"], sample_w_rewards["answerKey"])
        #print(validate_answer(sample_w_rewards["rewards"], sample_w_rewards["answerKey"]))
        data_w_rewards.append(sample_w_rewards)
    
    print("Number of samples: ", len(data))
    print(f"accuracy: {correct_cnt / len(data)})")
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
        default="../evals/datasets/commonsenseqa/train_rand_split.jsonl",
        help="Path to the dataset (HuggingFace path or local folder)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../evals/datasets/commonsenseqa/train_rand_split_w_rewards.jsonl",
        help="Path to the dataset (HuggingFace path or local folder)",
    )
    args = parser.parse_args()  # Parse the provided command-line arguments

    data_w_rewards = compute_dataset_rewards(args.data_path, args.model_path)

    with open(args.output_path, "w") as outf:
        for x in data_w_rewards:
            json.dump(x, outf)
            outf.write("\n")
