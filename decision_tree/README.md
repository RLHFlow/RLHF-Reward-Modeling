# Interpreting Language Model Preferences Through the Lens of Decision Trees

+ **Author** [Min Li](https://min-li.github.io/)
+ **Blog**: https://rlhflow.github.io/posts/2025-01-22-decision-tree-reward-model/
+ **Models**:
  + [**Decision-Tree-Reward-Gemma-2-27B**](https://huggingface.co/RLHFlow/Decision-Tree-Reward-Gemma-2-27B)
  + [**Decision-Tree-Reward-Llama-3.1-8B**](https://huggingface.co/RLHFlow/Decision-Tree-Reward-Llama-3.1-8B)
+ **Dataset**: [**LLM-Preferences-HelpSteer2**](https://huggingface.co/datasets/RLHFlow/LLM-Preferences-HelpSteer2)
+ **Code Repository:** https://github.com/RLHFlow/RLHF-Reward-Modeling/decision_tree/
+ **Tech Report**: To release soon

## RewardBench Leaderboard (Jan 2025)

Rank | Model | Base Model | Method | Overall Score | Chat     | Chat Hard | Safety   | Reasoning |
|:------|:------|:-----------|:-------|:--------------|:---------|:----------|:---------|:----------|
1 | [**Decision-Tree-Reward-Gemma-2-27B**](https://huggingface.co/RLHFlow/Decision-Tree-Reward-Gemma-2-27B) | Gemma-2-27B | Decision Tree | **95.4**      | 96.9     | **91.4**  | 93.9     | **99.2**  |
2 | INF-QRM-Llama3.1-70B | Llama-3.1-70B | Sequence Classifier | 95.1          | 96.6     | 91.0      | 93.6     | 99.1      |
3 | [**Decision-Tree-Reward-Llama-3.1-8B**](https://huggingface.co/RLHFlow/Decision-Tree-Reward-Llama-3.1-8B) | Llama-3.1-8B | Decision Tree | 94.5          | 96.6     | 89.5      | 93.2     | 98.6      |
4 | QRM-Gemma-2-27B | Gemma-2-27B | Sequence Classifier | 94.4          | 96.6     | 90.1      | 92.7     | 98.3      |
5 | Skywork-Reward-Gemma-2-27B-v0.2 | Gemma-2-27B | Sequence Classifier | 94.3          | 96.1     | 89.9      | 93.0     | 98.1      |
6 | Llama-3.1-Nemotron-70B-Reward | Llama-3.1-70B | Custom Classifier | 94.1          | 97.5     | 85.7      | **95.1** | 98.1      |
7 | Skywork-Reward-Gemma-2-27B | Gemma-2-27B | Sequence Classifier | 93.8          | 95.8     | **91.4**  | 91.9     | 96.1      |
8 | TextEval-Llama3.1-70B | Llama-3.1-70B | Generative | 93.5          | 94.1     | 90.1      | 93.2     | 96.4      |
9 | MetaMetrics-RM-v1.0 | - | Custom Classifier | 93.4          | **98.3** | 86.4      | 90.8     | 98.2      |
10 | Skywork-Critic-Llama-3.1-70B | Llama-3.1-70B | Generative | 93.3          | 96.6     | 87.9      | 93.1     | 95.5      |
11 | QRM-Llama3.1-8B-v2 | Llama-3.1-8B | Sequence Classifier | 93.1          | 96.4     | 86.8      | 92.6     | 96.8      |
12 | Skywork-Reward-Llama-3.1-8B-v0.2 | Llama-3.1-8B | Sequence Classifier | 93.1          | 94.7     | 88.4      | 92.7     | 96.7      |

## Reward Model Usage

Before using the model, ensure you have the following dependencies installed:
- `transformers==4.45.2`
- `torch>=2.5.0`
- `flash-attn>=2.6.3`

Note: This code requires a GPU with NVIDIA Ampere architecture or newer.
```python
from transformers import AutoModelForSequenceClassification
import torch
from transformers import AutoTokenizer
model_name = "Decision-Tree-Reward-Llama-3.1-8B" # Another choice is "Decision-Tree-Reward-Gemma-2-27B" 
repo_id = f"RLHFlow/{model_name}"
device = "cuda"
# Initialize the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(repo_id, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device)
tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
# Load the decision tree
model.load_decision_tree(repo_id, filename="decision_tree.pkl")

# Prompt and response pairs
prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

# Compare the two responses
output = model.compare(prompt, response1, response2, tokenizer, device)
print("Response 1 rewards")
print(dict(zip(output["attributes"], output["rewards"][0])))
# {'helpfulness': 3.9603815, 'correctness': 3.9727726, 'coherence': 3.8582935, 'complexity': 0.9909791, 'verbosity': 1.4901903}
print("Response 2 rewards")
print(dict(zip(output["attributes"], output["rewards"][1])))
# {'helpfulness': 2.1698856, 'correctness': 2.2035594, 'coherence': 3.2032843, 'complexity': 0.8786768, 'verbosity': 1.4569137}
print("Model preference")
print(output["preference"])
# 0

```
## LLM Preference Data Collection

**Script**: `collect_llm_preferences.py`

**Goal**: Collect LLM preferences on HelpSteer2 dataset using Together API, OpenAI API, Anthropic API or Gemini API.

**Usage**:
1. Run specific models:
```bash
   python collect_llm_preferences.py --api_key YOUR_API_KEY --vendor VENDOR --models MODEL1 MODEL2 [--split val/train] [--num_examples N] [--output_folder path]
   # Example: python collect_llm_preferences.py --api_key xxx --vendor together --models Mixtral-8x7B Llama-3-70B --split val --num_examples 5
```
2. Run all models:
```bash
   python collect_llm_preferences.py --api_key YOUR_API_KEY --vendor VENDOR --run_all [--split val/train] [--num_examples N] [--output_folder path]
   # Example: python collect_llm_preferences.py --api_key xxx --vendor together --run_all --split val --num_examples 10
```
**Arguments**:
* `--api_key`: API key for the selected vendor (can also be set via environment variable)
* `--vendor`: API vendor to use ('together', 'openai', or 'anthropic')
* `--models`: List of model names to evaluate (mutually exclusive with `--run_all`)
* `--run_all`: Run all non-problematic models (mutually exclusive with `--models`)
* `--output_folder`: Folder to save results (default: 'results')
* `--split`: Dataset split to use, either 'train' or 'val' (default: 'train')
* `--num_examples`: Number of examples to process (default: all)

## To-Do
+ [x] Reward Model Usage code
+ [x] Data Collection Code
+ [ ] Decision Tree Code
