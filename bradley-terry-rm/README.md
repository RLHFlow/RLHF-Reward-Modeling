# RLHF-Reward-Modeling: BT Model

## Installation instructions

The current solution is based on the alignment handbook and the environment, which should be sufficient for plain RM training.
Before starting, please make sure your linux machine has [nvidia-cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) installed.

```shell
conda create -n rm_dev python=3.10.9
conda activate rm_dev

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9
# The test cuda version is 12.1, 12.2. You may need to update the torch version based on your cuda version...
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install flash-attn
pip install accelerate==0.27.2
## You may need to modify the version of transformers for the model you want to use...
pip install transformers==4.39.0

git clone https://github.com/WeiXiongUST/RLHF-Reward-Modeling.git
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

## Running the Code

Running the code with Gemma-2b-it.

```shell
accelerate launch ./bradley-terry-rm/gemma_rm.py --model_name google/gemma-2b-it --max_length 4096 --train_set_path weqweasdas/preference_dataset_mix2
```

You can also modify the learning rate, batch size, output_path.. with either command or modify the ScriptArguments in the rm_gemma.py

If you encounter out-of-memory issue. Running the code with Gemma-2b-it with deepspeed stage 3. If OOM still exists, use a smaller max length and per_device_batch_size.

```shell
accelerate launch ./bradley-terry-rm/gemma_rm.py --model_name google/gemma-2b-it --max_length 4096 --train_set_path weqweasdas/preference_dataset_mix2 --deepspeed ./deepspeed_configs/deepspeed_3.json
```

**REMARK: note that with deepspeed stage 3, the final mode saving does not work normally. You should set the save_every_steps as the total number of training steps - 1 so that the trainer will save a model for you just before finishing the training.**


For the models without an official padding token (like Mistral and LLaMA3), you can run the mistral script and other parameters can be set in a similar way.

## Service the RM

Here is an example to use the RM.

```python
from transformers import AutoTokenizer, pipeline
rm_tokenizer = AutoTokenizer.from_pretrained("sfairXC/FsfairX-LLaMA3-RM-v0.1")
device = 0 # accelerator.device
rm_pipe = pipeline(
    "sentiment-analysis",
    model="sfairXC/FsfairX-LLaMA3-RM-v0.1",
    #device="auto",
    device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16}
)

pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1
}

chat = [
{"role": "user", "content": "Hello, how are you?"},
{"role": "assistant", "content": "I'm doing great. How can I help you today?"},
]
# You can prepare a list of texts like [text1, text2, ..., textn] and get rewards = [reward1, reward2, ..., rewardn]
test_texts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")]
pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
rewards = [output[0]["score"] for output in pipe_outputs]
```