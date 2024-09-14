from transformers import AutoTokenizer, pipeline
import torch
rm_tokenizer = AutoTokenizer.from_pretrained("/home/lichangc_google_com/RLHF-Reward-Modeling/bt_models/gemma2b_rm/last_checkpoint")
device = 0 # accelerator.device
rm_pipe = pipeline(
    "sentiment-analysis",
    model="/home/lichangc_google_com/RLHF-Reward-Modeling/bt_models/gemma2b_rm/last_checkpoint",
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
test_texts = [rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")]
pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
# breakpoint()
rewards = [output[1]["score"] for output in pipe_outputs] # head 1 is the length head while head 2 is the quality head.