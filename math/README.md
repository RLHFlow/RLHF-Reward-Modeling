# RLHF-Reward-Modeling: Math Reward

Model
- [RLHFlow/Llama3.1-8B-PRM-Mistral-Data](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Mistral-Data)
- [RLHFlow/Llama3.1-8B-ORM-Mistral-Data](https://huggingface.co/RLHFlow/Llama3.1-8B-ORM-Mistral-Data)
- [RLHFlow/Llama3.1-8B-PRM-Deepseek-Data](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Deepseek-Data)
- [RLHFlow/Llama3.1-8B-ORM-Deepseek-Data](https://huggingface.co/RLHFlow/Llama3.1-8B-ORM-Deepseek-Data)

## Introduction

We present an implementation of process-supervision reward model, as described in the [Math Shepherd paper](https://arxiv.org/abs/2312.08935). We follow the original paper to use the hard label and SFT training pipeline.

## Installation instructions

Before starting, please make sure your linux machine has [nvidia-cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) installed. The environment setup is the same as the pairwise preference model.

```shell
conda create -n prm_dev python=3.10.9
conda activate prm_dev

## Get axolotl for general model
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
git checkout 55cc214c767741e83ee7b346e5e13e6c03b7b9fa
pip install -e .

# The test cuda version is 12.1, 12.2. You may need to update the torch version based on your cuda version...
# you may encounter underfined symbol error related to cuda and flash-attn and 2.1.2 can solve it ...
pip3 install torch==2.1.2 torchvision torchaudio
pip install flash-attn


## Get FastChat
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .

git clone https://github.com/WeiXiongUST/RLHF-Reward-Modeling.git
pip install deepspeed
```

You also need to install wandb to record the training and log in with the huggingface accout to access Gemma.

```shell
pip install wandb
wandb login

huggingface-cli login
```

Some possible problems:

`CUDA_HOME` may not exist, unable to compile CUDA op(s)AssertionError:[end of output]

```shell
conda install nvidia/label/cuda-12.2.0::cuda-nvcc
```

## Dataset Preparation
The problem is formulated as a multi-turn chat and the data should be processed into the standard format. See [RLHFlow/Llama3.1-8B-PRM-Mistral-Data](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Mistral-Data) and [RLHFlow/Mistral-ORM-Data](https://huggingface.co/datasets/RLHFlow/Mistral-ORM-Data) for an example.

## Running the Code

Running the code for PRM/ORM.

```shell
torchrun --nproc_per_node 8 --master_port 20001 -m axolotl.cli.train llama-3.1-prm.yaml --deepspeed ../deepspeed_configs/deepspeed_3.json
```

## Usage Example for Pairwise Comparison

We provide test sets on GSM8K and MATH500 for best-of-N evaluation
- [RLHFlow/Mistral-GSM8K-Test](https://huggingface.co/datasets/RLHFlow/Mistral-GSM8K-Test)
- [RLHFlow/Mistral-MATH500-Test](https://huggingface.co/datasets/RLHFlow/Mistral-MATH500-Test)
- [RLHFlow/Deepseek-GSM8K-Test](https://huggingface.co/datasets/RLHFlow/Deepseek-GSM8K-Test)
- [RLHFlow/Deepseek-MATH500-Test](https://huggingface.co/datasets/RLHFlow/Deepseek-MATH500-Test)

### A sample of the dataset:
```
{
      "prompt": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates."
      "answers":
      [
            "To convert from rectangular coordinates $(x, y)$ to polar coordinates $(r, \\theta)$, we can use the formulas\n\\[r = \\sqrt{x^2 + y^2}\\]\n\\[\\theta = \\arctan \\frac{y}{x}\\]\n\nIn this case, the rectangular coordinates are $(0,3)$, so $x = 0$ and $y = 3$. \n\nFirst, we calculate $r$:\n\\[r = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3\\]\n\nNext, we calculate $\\theta$:\n\\[\\theta = \\arctan \\frac{3}{0}\\]\nSince the tangent function is not defined for $x = 0$, we need to use a special case. When $x = 0$, $\\theta = \\frac{\\pi}{2}$ if $y > 0$, and $\\theta = \\frac{3\\pi}{2}$ if $y < 0$. In this case, $y = 3 > 0$, so $\\theta = \\frac{\\pi}{2}$.\n\nSo, the polar coordinates equivalent to $(0,3)$ are $\\boxed{(3,\\frac{\\pi}{2})}$.",
            "The conversion from rectangular coordinates $(x,y)$ to polar coordinates $(r,\\theta)$ is given by the formulas $r = \\sqrt{x^2 + y^2}$ and $\\theta = \\arctan\\left(\\frac{y}{x}\\right)$.\n\nIn this case, the rectangular coordinates are $(x,y) = (0,3)$. Substituting these values into the formulas, we get $r = \\sqrt{0^2 + 3^2} = 3$ and $\\theta = \\arctan\\left(\\frac{3}{0}\\right)$. \n\nHowever, we need to be careful with the angle $\\theta$. The arctan function returns values in the range $-\\frac{\\pi}{2} < \\theta < \\frac{\\pi}{2}$, but we need $\\theta$ to be in the range $0 \\le \\theta < 2 \\pi$. \n\nFor the point $(0,3)$, the angle $\\theta$ is exactly $\\frac{\\pi}{2}$ (or 90 degrees) if we measure it counterclockwise from the positive x-axis. However, this angle is not in the range $0 \\le \\theta < 2 \\pi$. \n\nInstead, we can use the fact that a full circle is $2\\pi$ radians. So, an angle of $\\frac{\\pi}{2}$ is equivalent to an angle of $2\\pi - \\frac{\\pi}{2} = \\frac{3\\pi}{2}$. \n\nTherefore, the polar coordinates of the point $(0,3)$ are $\\boxed{(3,\\frac{3\\pi}{2})}$."
            ....
      ]
      "labels": [1,0,....]
}
```
The length of `answers` is the same as the length of `labels`.

each position of `labels` indicates whether that position of `answers` is correct (1 means correct and 0 means incorrect).

### Evaluation Methods:

**Process Reward Evaluation:**

Each answer will be split into steps and converted into the following multi-turn conversation format:
```
[
      {"role": "user", "content": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. To convert from rectangular coordinates $(x, y)$ to polar coordinates $(r, \\theta)$, we can use the formulas\n\\[r = \\sqrt{x^2 + y^2}\\]\n\\[\\theta = \\arctan \\frac{y}{x}\\]"},
      {"role": "assistant", "content": "+"},
      {"role": "user", "content": "In this case, the rectangular coordinates are $(0,3)$, so $x = 0$ and $y = 3$."},
      {"role": "assistant", "content": "+"},
      {"role": "user", "content": "First, we calculate $r$:\n\\[r = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3\\]"},
      {"role": "assistant", "content": "+"},
      {"role": "user", "content": "Next, we calculate $\\theta$:\n\\[\\theta = \\arctan \\frac{3}{0}\\]"},
      {"role": "assistant", "content": "+"},
      {"role": "user", "content": "Since the tangent function is not defined for $x = 0$, we need to use a special case. When $x = 0$, $\\theta = \\frac{\\pi}{2}$ if $y > 0$, and $\\theta = \\frac{3\\pi}{2}$ if $y < 0$."},
      {"role": "assistant", "content": "+"},
      {"role": "user", "content": "In this case, $y = 3 > 0$, so $\\theta = \\frac{\\pi}{2}$."},
      {"role": "assistant", "content": "+"},
      {"role": "user", "content": "So, the polar coordinates equivalent to $(0,3)$ are $\\boxed{(3,\\frac{\\pi}{2})}$."},
      {"role": "assistant", "content": "+"}, 
]
```
We extract the probability of `+` from the assistant in each turn. It represents the process reward score for each step.

We calculate the average of the process reward score, as the final representation of the answer quality.

**Outcome Reward Evaluation:**

Each answer will be converted directly into the following single-turn conversation format:
```
[
      {"role": "user", "content": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. To convert from rectangular coordinates $(x, y)$ to polar coordinates $(r, \\theta)$, we can use the formulas\n\\[r = \\sqrt{x^2 + y^2}\\]\n\\[\\theta = \\arctan \\frac{y}{x}\\]\n\nIn this case, the rectangular coordinates are $(0,3)$, so $x = 0$ and $y = 3$. \n\nFirst, we calculate $r$:\n\\[r = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3\\]\n\nNext, we calculate $\\theta$:\n\\[\\theta = \\arctan \\frac{3}{0}\\]\nSince the tangent function is not defined for $x = 0$, we need to use a special case. When $x = 0$, $\\theta = \\frac{\\pi}{2}$ if $y > 0$, and $\\theta = \\frac{3\\pi}{2}$ if $y < 0$. In this case, $y = 3 > 0$, so $\\theta = \\frac{\\pi}{2}$.\n\nSo, the polar coordinates equivalent to $(0,3)$ are $\\boxed{(3,\\frac{\\pi}{2})}$."},
      {"role": "assistant", "content": "+"},
]
```
We extract the probability of `+` from the assistant. It represents the outcome reward score for this answer.

**Answer Selection:**

Once we get the reward scores for a pair or more answers, we select the answer with the highest score.

We compare the answer with the corresponding label to see whether the selected one is correct or not.

### Running the evaluation code:
```
accelerate launch prm_evaluate.py --reward_name_or_path RLHFlow/Llama3.1-8B-PRM-Mistral-Data \
      --dataset RLHFlow/Mistral-MATH500-Test \
      --output_dir Mistral-PRM-Mistral-MATH500 \
      --num_n 1024 \
      --model_type Mistral
```
You may specify `--reward_name_or_path` as a local model or Huggingface model, `--dataset` from above, or your own with the same format. 

The results will be stored in `{output_dir}.json`. 

The `--num_n` specifies the N for best-of-N. 

The `--model_type` should be either `Mistral` or `Deepseek`, depending on the training data.

You may replace `prm_evaluate.py` with `orm_evaluate.py` if you want to evaluate the Outcome Reward Model.

# Experiment Setup

# Experiment Results

```bibtex
@misc{xiong2024rlhflowmath,
      author={Wei Xiong and Hanning Zhang and Tong Zhang},
  title = {An Implementation of Generative PRM},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/RLHFlow/RLHF-Reward-Modeling}}
}
```
