# RLHF-Reward-Modeling

TL;DL: this is a repo for training the reward model for [RLHF (PPO)](https://arxiv.org/pdf/2203.02155.pdf), [rejection sampling fine-tuning](https://arxiv.org/pdf/2304.06767v4.pdf), and [iterative DPO](https://arxiv.org/pdf/2312.11456.pdf).

- 4 x A40 48G: we can train Gemma-7B-it with max_length 4096 with deepspeed3 + gradient checkpoint;
- 4 x A100 80G: we can train Gemma-7B-it with max_length 4096 with gradient checkpoint.

## Installation instructions

To do

One tentative choice is to install the alignment handbbok and the environment should be sufficient.

```shell
conda create -n newhandbook python=3.10.9
conda activate newhandbook

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .

python -m pip install .
pip install flash-attn

git clone https://github.com/WeiXiongUST/RLHF-Reward-Modeling.git
```

Some possible problems:

CUDA HOME does not exist, unable to compile CUDA op(s)AssertionError:[end of output]

```shell
conda install nvidia/label/cuda-12.2.0::cuda-nvcc
```

You also need to install wandb to record the training and log in with the huggingface accout to access Gemma.

```shell
pip install wandb
wandb login

huggingface-cli login
```

## Dataset Preparation
The dataset should be preprocessed as the standard format, where each of the sample consists of two conversations 'chosen' and 'rejected' and they share the same prompt. Here is an example of the rejected sample in the comparison pair. 

```python
[
{ "content": "Please identify the top 5 rarest animals in the world.", "role": "user" },
{ "content": "Do you mean animals that are really rare, or rare relative to the size of the human population?", "role": "assistant" },
{ "content": "The ones that are really rare.", "role": "user" },
{"content": "Alright, here’s what I found:", "role": "assistant" } ]
```

We preprocess 4 dataset and upload them to the hugginface hub. 

- Version 1: [weqweasdas/preference_dataset_mixture](weqweasdas/preference_dataset_mixture)
- Version 2: [weqweasdas/preference_dataset_mix2](weqweasdas/preference_dataset_mix2)
- Version 3: [weqweasdas/preference_dataset_mixture2_and_safe_pku](weqweasdas/preference_dataset_mixture2_and_safe_pku)
- Version 4: [weqweasdas/preference_dataset_mixture2_and_safe_pku150k](weqweasdas/preference_dataset_mixture2_and_safe_pku150k)

**Version 1:** The model is trained on a **mixture1** of

- [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [SHP](https://huggingface.co/datasets/stanfordnlp/SHP)
- [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)
- [Capybara](https://www.notion.so/argilla/distilabel-capybara-dpo-7k-binarized)
- [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer)
- [Orca](https://www.notion.so/argilla/distilabel-intel-orca-dpo-pairs)

The total number of the comparison pairs is 250K, where we perform the following data selection and cleaning strateges:

- HH-RLHF: we use all the base, rejection sampling, and online subsets but delete the samples whose chosen == rejected, leading to 115547;
- SHP: we only use the samples with score ratio > 2, for each prompt, we only take 1 comparison, leading to 55916;
- Ultrafeedback: similar to [UltraFeedback-Binarized](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned), we use the fine-grained score instead of the overall one to rank samples. Meanwhile, for each prompt, we take the best one v.s. random chosen one in the remaining samples. Finally, we delete the selected pairs with equal scores, leading to 62793.
- HelpSteer: we use the mean of helpfulness and correctness to rank samples. Meanwhile, we take the best sample v.s. the random chosen one in the remaining samples. Finally, we delete the selected pairs with equal scores, leading to 8206;
- Capybara: we delete the pairs whose chosen and rejected samples are of the same rating, leading to 7562;
- Orca: we delete the pairs whose chosen and rejected samples are of the same rating, leading to 6405.

**Version 2:** The model is also trained on a **mixture2** of

- [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [SHP](https://huggingface.co/datasets/stanfordnlp/SHP)
- [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)
- [Capybara](https://www.notion.so/argilla/distilabel-capybara-dpo-7k-binarized)
- [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer)
- [Orca](https://www.notion.so/argilla/distilabel-intel-orca-dpo-pairs)

Difference:

- SHP: we only use the samples with score ratio > 2, for each prompt, we take 5 comparison at most, leading to 109526;
- Ultrafeedback: similar to [UltraFeedback-Binarized](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned), we use the fine-grained score instead of the overall one to rank samples. Meanwhile, for each prompt, we take all possible 6 pairs of comparisons. Finally, we delete the selected pairs with equal scores, leading to 267416.
- HelpSteer: we use the mean of helpfulness and correctness to rank samples. Meanwhile, we take all possible 6 pairs of comparisons. Finally, we delete the selected pairs with equal scores, leading to 21576;

**Version 3:** Mixture2 + 30K safety is the mixture2 + the training set of [PKU-Alignment/PKU-SafeRLHF-30K](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-30K)

**Version 4:** 1 Mixture2 + 150K safety is the mixture2 + 150K samples from [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)

## Running the Code


Running the code with Gemma-2b-it. 

```shell
accelerate launch rm.py --model_name google/gemma-2b-it --max_length 4096 --train_set_path weqweasdas/preference_dataset_mix2
```

You can also modify the learning rate, batch size, output_path.. with either command or modify the ScriptArguments in the rm_gemma.py

If you encounter out-of-memory issue. Running the code with Gemma-2b-it with deepspeed stage 3. If OOM still exists, use a smaller max length and per_device_batch_size.

```shell
accelerate launch rm.py --model_name google/gemma-2b-it --max_length 4096 --train_set_path weqweasdas/preference_dataset_mix2 --deepspeed deepspeed_3.json
```

**REMARK: note that with deepspeed stage 3, the final mode saving does not work normally. You should set the save_every_steps as the total number of training steps - 1 so that the trainer will save a model for you just before finishing the training.**

## Evaluation Results

You can evaluate the resulting reward model with the [benchmark](https://huggingface.co/datasets/allenai/reward-bench) by the following command and the result will be genera

```shell
accelerate launch eval_bench_mark.py --reward_name_or_path ./models/gemma_2b_mixture2_last_checkpoint --record_dir ./models/bench_mark_eval.txt
```

## Citation

The repo was part of the iterative rejection sampling fine-tuning and iterative DPO. If you find the content of this repo useful in your work, please consider cite it as follows:

```bibtex
@article{dong2023raft,
  title={Raft: Reward ranked finetuning for generative foundation model alignment},
  author={Dong, Hanze and Xiong, Wei and Goyal, Deepanshu and Pan, Rui and Diao, Shizhe and Zhang, Jipeng and Shum, Kashun and Zhang, Tong},
  journal={arXiv preprint arXiv:2304.06767},
  year={2023}
}

@article{xiong2023gibbs,
  title={Gibbs sampling from human feedback: A provable kl-constrained framework for rlhf},
  author={Xiong, Wei and Dong, Hanze and Ye, Chenlu and Zhong, Han and Jiang, Nan and Zhang, Tong},
  journal={arXiv preprint arXiv:2312.11456},
  year={2023}
}
```
