# RLHF-Reward-Modeling


## Installation instructions

To do

## Running the Code

**Dataset preparation**


Running the code with Gemma-2b-it. 

```shell
accelerate launch rm.py --model_name google/gemma-2b-it --max_length 4096 --train_set_path weqweasdas/preference_dataset_mix2
```

You can also modify the learning rate, batch size, output_path.. with either command or modify the ScriptArguments in the rm_gemma.py

If you encounter out-of-memory issue. Running the code with Gemma-2b-it with deepspeed stage 3. If OOM still exists, use a smaller max length and per_device_batch_size.

```shell
accelerate launch rm.py --model_name google/gemma-2b-it --max_length 4096 --train_set_path weqweasdas/preference_dataset_mix2 --deepspeed deepspeed_3.json
```
