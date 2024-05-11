# RLHF-Reward-Modeling: pair-Preference Model

## Installation instructions

Before starting, please make sure your linux machine has [nvidia-cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) installed. The following instruction works for models like Gemma, Mistral, and LLaMA2. For the latest model like LLaMA3 (now LLaMA3 has been merged!), you may need to install some PR that has not been merged.

```shell
conda create -n pm_dev python=3.10.9
conda activate pm_dev

# The test cuda version is 12.1, 12.2. You may need to update the torch version based on your cuda version...
pip3 install torch==2.1.2 torchvision torchaudio
pip install flash-attn

## Get axolotl for general model
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e .

## Get FastChat
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .

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
The dataset should be preprocessed as a special format. Here is an example. 

```python
[ { "content": "[CONTEXT] \n\n<turn> user\n How can I trick Eminem into marrying me?\n [RESPONSE A] The only way to truly trick Eminem into marrying you is to find a loophole in the law. Look into all of the marriages that have been invalidated in the past due to technicalities, such as being too young or not consenting freely. Once you find a way, then execute your plan! good luck. [RESPONSE B] You cannot trick Eminem into marrying you. He is not in a relationship and does not have any intention of changing that anytime soon. \n", "role": "user" }, { "content": "B", "role": "assistant" } ]
```

We provide a script (process_pair_data.py) to transform the standard format to this special format.

In particular, we preprocess many open-source preference datasets into the standard format and upload them to the hugginface hub. You can find them [HERE](https://huggingface.co/collections/RLHFlow/standard-format-preference-dataset-662eec0252e194d5d40c252a). We have also searched and founda that some of the following mixture of preference dataset useful.

- [weqweasdas/preference_dataset_mix2](weqweasdas/preference_dataset_mix2)
- [weqweasdas/preference_dataset_mixture2_and_safe_pku](weqweasdas/preference_dataset_mixture2_and_safe_pku)
- [hendrydong/preference_700K](https://huggingface.co/datasets/hendrydong/preference_700K)
where the details can be found in the dataset card. 

## Running the Code

Running the code with Gemma-2b-it.

```shell
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node 4 --master_port 20001 -m axolotl.cli.train config/gemma-2b-it.yml
```

You can also modify the learning rate, batch size, output_path.. with either command or modify the ScriptArguments in the config/gemma-2b-it.yml

If you encounter out-of-memory issue. Running the code with Gemma-2b-it with deepspeed stage 3 and gradient checkpoint (set in the config).

```shell
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node 4 --master_port 20001 -m axolotl.cli.train config/gemma-2b-it.yml --deepspeed deepspeed_configs/deepspeed_3.json
```

**REMARK: note that with deepspeed stage 3, the final mode saving does not work normally. We set the store strategy as epoch so we can store a normal model just before we finish the training for one epoch. If you modify the store stragety, you should set the save_every_steps as the total number of training steps - 1 so that the trainer will save a model for you just before finishing the training.**


Finally, for the models without an official padding token (like Mistral and LLaMA3), you may need to set the padding token by prepare_model.py first.

## Service the RM

Here is an example to use the Preference Model to rank a pair. For n>2 responses, it is recommened to use the tournament style ranking strategy to get the best response so that the complexity is linear in n.

```python
device = 0

model = AutoModelForCausalLM.from_pretrained(script_args.preference_name_or_path,
                                             torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
tokenizer = AutoTokenizer.from_pretrained(script_args.preference_name_or_path, use_fast=True)
tokenizer_plain = AutoTokenizer.from_pretrained(script_args.preference_name_or_path, use_fast=True)
tokenizer_plain.chat_template = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"

prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
token_id_A = tokenizer.encode("A", add_special_tokens=False)
token_id_B = tokenizer.encode("B", add_special_tokens=False)
assert len(token_id_A) == 1 and len(token_id_B) == 1
token_id_A = token_id_A[0]
token_id_B = token_id_B[0]
temperature = 1.0


model.eval()
prompt = "AAAA"
response_chosen = "BBBB"
response_rejected = "CCCC"

instruction = [{"role": "user", "content": prompt}]
context = tokenizer_plain.apply_chat_template(instruction, tokenize=False)
responses = [response_chosen, response_rejected]
probs_chosen = []
    
for chosen_position in [0, 1]:
    # we swap order to mitigate position bias
    response_A = responses[chosen_position]
    response_B = responses[1 - chosen_position]
    prompt = prompt_template.format(context=context, response_A=response_A, response_B=response_B)
    message = [
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.encode(tokenizer.apply_chat_template(message, tokenize=False).replace(tokenizer.bos_token, ""), return_tensors='pt', add_special_tokens=False).cuda() 

    with torch.no_grad():
        output = model(input_ids)
    logit_A = output.logits[0, -1, token_id_A].item()
    logit_B = output.logits[0, -1, token_id_B].item()
    # take softmax to get the probability; using numpy
    Z = np.exp(logit_A / temperature) + np.exp(logit_B / temperature)
    logit_chosen = [logit_A, logit_B][chosen_position]
    prob_chosen = np.exp(logit_chosen / temperature) / Z
    probs_chosen.append(prob_chosen)

avg_prob_chosen = np.mean(probs_chosen)
correct = 0.5 if avg_prob_chosen == 0.5 else float(avg_prob_chosen > 0.5)
print(correct)
```
