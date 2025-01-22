# Interpreting Language Model Preferences Through the Lens of Decision Trees

+ **Author** [Min Li](https://min-li.github.io/)
+ **Blog**: https://rlhflow.github.io/posts/2025-01-22-decision-tree-reward-model/
+ **Models**:
  + [**Decision-Tree-Reward-Gemma-2-27B**](https://huggingface.co/RLHFlow/Decision-Tree-Reward-Gemma-2-27B)
  + [**Decision-Tree-Reward-Llama-3.1-8B**](https://huggingface.co/RLHFlow/Decision-Tree-Reward-Llama-3.1-8B)
+ **Code Repository:** https://github.com/RLHFlow/RLHF-Reward-Modeling/decision_tree/
+ **Tech Report**: To release soon

## RewardBench Leaderboard (Jan 2025)

Rank | Model | Base Model | Method | Overall Score | Chat | Chat Hard | Safety | Reasoning |
|:------|:------|:-----------|:-------|:------|:-----|:----------|:-------|:----------|
1 | [**Decision-Tree-Reward-Gemma-2-27B**](https://huggingface.co/RLHFlow/Decision-Tree-Reward-Gemma-2-27B) | Gemma-2-27B | Decision Tree | **95.3** | 96.9 | **91.4** | 93.7 | **99.1** |
2 | INF-QRM-Llama3.1-70B | Llama-3.1-70B | Sequence Classifier | 95.1 | 96.6 | 91.0 | 93.6 | **99.1** |
3 | QRM-Gemma-2-27B | Gemma-2-27B | Sequence Classifier | 94.4 | 96.6 | 90.1 | 92.7 | 98.3 |
4 | Skywork-Reward-Gemma-2-27B-v0.2 | Gemma-2-27B | Sequence Classifier | 94.3 | 96.1 | 89.9 | 93.0 | 98.1 |
5 | [**Decision-Tree-Reward-Llama-3.1-8B**](https://huggingface.co/RLHFlow/Decision-Tree-Reward-Llama-3.1-8B) | Llama-3.1-8B | Decision Tree | 94.3 | 96.9 | 89.3 | 92.9 | 98.5 |
6 | Llama-3.1-Nemotron-70B-Reward | Llama-3.1-70B | Custom Classifier | 94.1 | 97.5 | 85.7 | **95.1** | 98.1 |
7 | Skywork-Reward-Gemma-2-27B | Gemma-2-27B | Sequence Classifier | 93.8 | 95.8 | **91.4** | 91.9 | 96.1 |
8 | TextEval-Llama3.1-70B | Llama-3.1-70B | Generative | 93.5 | 94.1 | 90.1 | 93.2 | 96.4 |
9 | MetaMetrics-RM-v1.0 | - | Custom Classifier | 93.4 | **98.3** | 86.4 | 90.8 | 98.2 |
10 | Skywork-Critic-Llama-3.1-70B | Llama-3.1-70B | Generative | 93.3 | 96.6 | 87.9 | 93.1 | 95.5 |
11 | QRM-Llama3.1-8B-v2 | Llama-3.1-8B | Sequence Classifier | 93.1 | 96.4 | 86.8 | 92.6 | 96.8 |
12 | Skywork-Reward-Llama-3.1-8B-v0.2 | Llama-3.1-8B | Sequence Classifier | 93.1 | 94.7 | 88.4 | 92.7 | 96.7 |


## To-Do
+ [ ] Reward Model Usage code
+ [ ] Data Collection Code
+ [ ] Decision Tree Code
