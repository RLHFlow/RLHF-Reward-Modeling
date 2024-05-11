# RLHF-Reward-Modeling

🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥

🔥🔥🚀🚀🚀 **The pairwise preference model training is available now!** 🚀🚀🚀🔥🔥

🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥


<img width="625" alt="image" src="https://github.com/RLHFlow/RLHF-Reward-Modeling/assets/90632760/bf5184c9-0c06-464b-8f68-cb86d71eab25">


TL;DL: this is a repo for training the reward/preference model for [DRL-based RLHF (PPO)](https://arxiv.org/pdf/2203.02155.pdf), [Iterative SFT (Rejection sampling fine-tuning)](https://arxiv.org/pdf/2304.06767v4.pdf), and [iterative DPO](https://arxiv.org/pdf/2312.11456.pdf).

- 4 x A40 48G: we can train Gemma-7B-it with max_length 4096 by Deepspeed Zero-3 + gradient checkpoint;
- 4 x A100 80G: we can train Gemma-7B-it with max_length 4096 by gradient checkpoint;
- The resulting reward models achieve **SOTA performance** as open-source RMs in the leaderboard of [RewardBench](https://huggingface.co/spaces/allenai/reward-bench).
- Check out our [blog post](https://efficient-unicorn-451.notion.site/Reward-Modeling-for-RLHF-abe03f9afdac42b9a5bee746844518d0)!



## Installation instructions

It is recommeded to create separate environmnets for the Bradley-Terry reward model and pair wise preference model. The installation instructions are provided in the corresponding folders.


## Dataset Preparation
The dataset should be preprocessed as the standard format, where each of the sample consists of two conversations 'chosen' and 'rejected' and they share the same prompt. Here is an example of the rejected sample in the comparison pair. 

```python
[
{ "content": "Please identify the top 5 rarest animals in the world.", "role": "user" },
{ "content": "Do you mean animals that are really rare, or rare relative to the size of the human population?", "role": "assistant" },
{ "content": "The ones that are really rare.", "role": "user" },
{ "content": "Alright, here’s what I found:", "role": "assistant" }, 
]
```

We preprocess many open-source preference datasets into the standard format and upload them to the hugginface hub. You can find them [HERE](https://huggingface.co/collections/RLHFlow/standard-format-preference-dataset-662eec0252e194d5d40c252a). We have also searched and founda that some of the following mixture of preference dataset useful.

- [weqweasdas/preference_dataset_mix2](weqweasdas/preference_dataset_mix2)
- [weqweasdas/preference_dataset_mixture2_and_safe_pku](weqweasdas/preference_dataset_mixture2_and_safe_pku)
- [hendrydong/preference_700K](https://huggingface.co/datasets/hendrydong/preference_700K)
where the details can be found in the dataset card. 

## Evaluation Results

You can evaluate the resulting reward model with the dataset provided by [benchmark](https://huggingface.co/datasets/allenai/reward-bench) by the following command.

```shell
CUDA_VISIBLE_DEVICES=1 python ./useful_code/eval_reward_bench_bt.py --reward_name_or_path ./models/gemma_2b_mixture2_last_checkpoint --record_dir ./bench_mark_eval.txt
```

Some models trained by our script are competitive in the leaderboard. 

![image](https://github.com/WeiXiongUST/RLHF-Reward-Modeling/assets/90632760/49f1663d-4dbb-4513-80cb-3fab00f6f955)


## To Do

- [x]  Bradley-Terry Reward Model based on Gemma and Mistral.
- [ ]  Bradley-Terry Reward Model based on Mixtral;
- [x]  Preference model;
- [ ]  Regression-based reward model;
- [ ]  Multi-objective reward model.

## Citation

The repo was part of the iterative rejection sampling fine-tuning and iterative DPO. If you find the content of this repo useful in your work, please consider cite it as follows:

```bibtex
@article{dong2023raft,
  title={Raft: Reward ranked finetuning for generative foundation model alignment},
  author={Dong, Hanze and Xiong, Wei and Goyal, Deepanshu and Pan, Rui and Diao, Shizhe and Zhang, Jipeng and Shum, Kashun and Zhang, Tong},
  journal={arXiv preprint arXiv:2304.06767},
  year={2023}
}

@misc{xiong2024iterative,
      title={Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint}, 
      author={Wei Xiong and Hanze Dong and Chenlu Ye and Ziqi Wang and Han Zhong and Heng Ji and Nan Jiang and Tong Zhang},
      year={2024},
      eprint={2312.11456},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
