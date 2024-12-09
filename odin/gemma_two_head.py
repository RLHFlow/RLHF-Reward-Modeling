########################
# This script is modified from the TRL package https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
# This script is designed for the reward modeling with Gemma model but can also be applied to any models with a chat template and an official pad token
# If you have any question, feel free to send me an email via wx13@illinois.edu
########################
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
# add a tiktoken
import tiktoken
import torch.distributed as dist
import torch.nn.functional as F
encoding = tiktoken.get_encoding("o200k_base")
# set the wandb offline
# import wandb
# wandb.init(mode='disabled')

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})

    deepspeed: Optional[str] = field(
        # default="dp3.json",
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )

    per_device_train_batch_size: Optional[int] = field(default=16)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=2)
    learning_rate: Optional[float] = field(default=2e-6)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="google/gemma-2b-it", #"mistralai/Mistral-7B-Instruct-v0.2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_set_path: Optional[str] = field(
        default="hendrydong/preference_700K",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    eval_set_path: Optional[str] = field(
        default="hendrydong/preference_700K",
        metadata={"help": "The dir of the subset of the eval data to use"},
    )
    output_path: Optional[str] = field(
        default="./bt_models/gemma2b_rm",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)

    save_every_steps: Optional[int] = field(
        default=1366,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Eval the model every x steps"},
    )
    correlation_with_length: Optional[float] = field(
        default=1.0,
        metadata={"help": "The weight of the length correlation loss"},
    )
    ortho_reg: Optional[float] = field(
        default=1.0,
        metadata={"help": "The weight of the orthogonal regularization"},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the value-head model and tokenizer.
tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

# Adjusted according to the base model
# Need to do this for the models that don't have an official pad token.
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length
correlation_with_length = script_args.correlation_with_length
ortho_reg = script_args.ortho_reg
# Get the dataset
train_path = script_args.train_set_path
eval_path = script_args.eval_set_path
output_name = script_args.output_path

def build_dataset(tokenizer, train_path, eval_path):

    def tokenize(sample):
        
        sample['positive'] = tokenizer.apply_chat_template(
            sample['chosen'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        sample['negative'] = tokenizer.apply_chat_template(
            sample['rejected'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        
        tokenized_pos = tokenizer(sample['positive'], truncation=True)
        tokenized_neg = tokenizer(sample['negative'], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        return sample
    #ds = load_dataset(train_path, split="train").shuffle(seed=42)
    # to have a quicker iteration, we just use 500 examples here.
    ds = load_dataset(train_path, split="train").shuffle(seed=42)
    #ds = ds.select(range(2000))
    ds = ds.map(tokenize, num_proc=16)

    eval_dataset = None

    train_dataset = ds
    eval_dataset = load_dataset(eval_path, split="train").shuffle(seed=42).select(range(500))
    #eval_dataset = ds.select(range(500))
    return train_dataset, eval_dataset


train_dataset, eval_dataset = build_dataset(tokenizer, train_path, eval_path)
print("Training set: ", len(train_dataset), " Eval set: ", len(eval_dataset))

# Define the trainer
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_strategy="steps",
    save_steps=script_args.save_every_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to='wandb'
)

# enable if you want to train with lora
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS,
#     inference_mode=False,
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
# )

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=2, torch_dtype=torch.bfloat16, use_flash_attention_2=True,
)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

model.config.use_cache = not script_args.gradient_checkpointing
num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        seqlens = []
        for feature in features:
            length_j = len(encoding.encode(feature['chosen'][1]['content']))
            length_k = len(encoding.encode(feature['rejected'][1]['content']))
            seqlens.append(length_j)
            seqlens.append(length_k)
            merged_features.append(
                {
                    # calculate the token length of the string using tiktoken
                    # "chosen_response_length": len(encoding.encode(feature['chosen'][1]['content'])),
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            merged_features.append(
                {
                    # "chosen_response_length": len(encoding.encode(feature['rejected'][1]['content'])),
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # breakpoint()
        batch = {
            "seqlens": seqlens,
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        # breakpoint()
        return batch


# Define the trainer
def compute_metrics(eval_pred):
    result = {}
    predictions = eval_pred.predictions
    # Extract scores from the second head (index 1)
    pos_predictions_scores = predictions[0][:, 1]
    neg_predictions_scores = predictions[1][:, 1]
    # We assume that the first sample is preferred by default in groundtruth
    result['accuracy'] = np.sum(
        pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
    return result


class RewardTrainer(Trainer):
    def proj_with_normalized_weight(self, weight):
        w = weight - torch.mean(weight, dim=-1, keepdim=True)
        head_norms = torch.norm(w, dim=-1, keepdim=True)
        head_norms = torch.maximum(head_norms, torch.full_like(head_norms, 1e-8))
        normalized_w = w / head_norms
        # Return the weight for orthogonal regularizations
        return normalized_w

    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        seqlens = inputs["seqlens"]
        seqlens = torch.tensor(seqlens, device=rewards.device)
        # do the all_gather here to get the seqlens from other gpus

        local_length_list = seqlens.contiguous()
        local_rewards_list = rewards.contiguous()

        all_length_list = [torch.zeros_like(local_length_list) for _ in range(dist.get_world_size())]
        all_rewards_list = [torch.zeros_like(local_rewards_list) for _ in range(dist.get_world_size())]
        dist.all_gather(all_length_list, local_length_list)
        dist.all_gather(all_rewards_list, local_rewards_list)

        # Replace the local device's data with the one that has gradients
        all_rewards_list[dist.get_rank()] = local_rewards_list
        
        # Concatenate all gathered tensors
        all_length_tensor = torch.cat(all_length_list, dim=0).to(rewards.device)
        all_rewards_tensor = torch.cat(all_rewards_list, dim=0).to(rewards.device)

        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx] # chosen response rewards
        rewards_k = rewards[kidx] # rejected response rewards
        ranking_loss = -nn.functional.logsigmoid(rewards_j.sum() - rewards_k.sum()).mean()
        
        # Length correlation loss for head 1 (encouraging correlation)
        length_corr_matrix1 = torch.stack((all_length_tensor, all_rewards_tensor[:, 0]))
        length_corr1 = torch.corrcoef(length_corr_matrix1.to(dtype=torch.float32))[0, 1]
        length_loss1 = 1 - length_corr1  # Encourage correlation

        # Length correlation loss for head 2 (discouraging correlation)
        length_corr_matrix2 = torch.stack((all_length_tensor, all_rewards_tensor[:, 1]))
        length_corr2 = torch.corrcoef(length_corr_matrix2.to(dtype=torch.float32))[0, 1]
        length_loss2 = torch.abs(length_corr2)  # Discourage correlation

        # the orthogonal loss: may have potential bugs when the zero-stage=3
        ortho_loss = 0
        # linear_layer = model.module.score
        # weight = linear_layer.weight
        # w_normalized = self.proj_with_normalized_weight(weight)

        # # Add orthogonal regularization on the projection layer weights
        # prod = w_normalized @ w_normalized.T

        # # Ensure prod is at least 2D
        # mean_corr = torch.abs(torch.triu(prod, diagonal=1)).sum()
        # ortho_loss = ortho_reg * mean_corr
        
        # Combine losses
        total_loss = ranking_loss + correlation_with_length * (length_loss1 + length_loss2) + ortho_loss
        # if torch.distributed.get_rank() == 0:
        #     breakpoint()
        
        if return_outputs:
            return total_loss, {
                "loss": total_loss,
                "length_loss1": length_loss1,
                "length_loss2": length_loss2,
                "ranking_corr_loss": ranking_loss,
                "rewards_j": rewards_j,
                "rewards_k": rewards_k
            }
        return total_loss


# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length),
)


trainer.train()


print("Saving last checkpoint of the model")
#model.save_pretrained(output_name + "/last_checkpoint")
trainer.save_model(output_name + "/last_checkpoint")
tokenizer.save_pretrained(output_name + "/last_checkpoint")
