from transformers import AutoTokenizer, pipeline, AutoModel
from huggingface_hub import delete_repo
import torch
model_path = "/home/lichangc_google_com/RLHF-Reward-Modeling/gemma_9b_700K/last_checkpoint"
# rm_tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
#upload the model to the huggingface hub
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
rm_tokenizer = AutoTokenizer.from_pretrained(model_path)

device = 0 # accelerator.device
rm_pipe = pipeline(
    "sentiment-analysis",
    model=model_path,
    #device="auto",
    device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16}
)
# upload the model to the huggingface
# !transformers-cli login
# !transformers-cli repo create gemma_9b_700K
# !transformers-cli repo upload

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
test_texts = [rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")]
pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
# breakpoint()
rewards = [output[1]["score"] for output in pipe_outputs] # head 1 is the length head while head 2 is the quality head.

print(rewards)