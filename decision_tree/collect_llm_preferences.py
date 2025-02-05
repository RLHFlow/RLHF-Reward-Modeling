"""
Script to collect LLM preferences on HelpSteer2 dataset using Together API, OpenAI API, Anthropic API or Gemini API.

Usage:
    1. Run specific models:
       python collect_llm_preferences.py --api_key YOUR_API_KEY --vendor VENDOR --models MODEL1 MODEL2 [--split val/train] [--num_examples N] [--output_folder path]
       Example: python collect_llm_preferences.py --api_key xxx --vendor together --models Mixtral-8x7B Llama-3-70B --split val --num_examples 5

    2. Run all non-problematic models:
       python collect_llm_preferences.py --api_key YOUR_API_KEY --vendor VENDOR --run_all [--split val/train] [--num_examples N] [--output_folder path]
       Example: python collect_llm_preferences.py --api_key xxx --vendor together --run_all --split val --num_examples 10

Arguments:
    --api_key: API key for the selected vendor (can also be set via environment variable)
    --vendor: API vendor to use ('together', 'openai', or 'anthropic')
    --models: List of model names to evaluate (mutually exclusive with --run_all)
    --run_all: Run all non-problematic models (mutually exclusive with --models)
    --output_folder: Folder to save results (default: 'results')
    --split: Dataset split to use, either 'train' or 'val' (default: 'train')
    --num_examples: Number of examples to process (default: all)
"""

import os
import time
import argparse
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from together import Together
from openai import OpenAI
import anthropic
from anthropic import Anthropic
import google.generativeai as genai

# Template for formatting conversations
TEMPLATE = """Here is a conversation between a user and an AI assistant. Each user's message begins with "<|USER|>" and each assistant's message begins with "<|ASSISTANT|>".
Below is the conversation history and two possible responses. Please indicate which response you think is better and explain why.
State your preference by starting with either "A is the better response" or "B is the better response", then explain your reasoning.

### Conversation
{conversation}

### Response A
{response_1}

### Response B
{response_2}"""

# exclude_models = ["WizardLM2-8x22B","DBRX-Instruct", "MythoMax-L2-13B"] # Does not follow the prompt well

# Model name mappings for different vendors
anthropic_model_strings = {
    "Claude-3.5-Sonnet": "claude-3-5-sonnet-20241022",
    "Claude-3.5-Haiku": "claude-3-5-haiku-20241022",
}

openai_model_strings = {
    "GPT-4-Turbo": "gpt-4-turbo-2024-04-09",
    "GPT-4o": "gpt-4o-2024-11-20",
    "GPT-4o-Mini": "gpt-4o-mini-2024-07-18",
}

gemini_model_strings = {
    "Gemini-1.5-Flash-8B": "gemini-1.5-flash-8b",
    "Gemini-1.5-Flash": "gemini-1.5-flash",
    "Gemini-1.5-Pro": "gemini-1.5-pro",
    "Gemini-1.0-Pro": "gemini-1.0-pro",
    "Gemini-2.0-Flash-Exp": "gemini-2.0-flash-exp",
}

# Original Together model mappings
together_api_model_strings = {
    "Llama-2-13B": "meta-llama/Llama-2-13b-chat-hf",
    "Llama-3-8B": "meta-llama/Llama-3-8b-chat-hf",
    "Llama-3-70B": "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
    "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "Llama-3.1-70B": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "Llama-3.1-405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "Llama-3.2-11B-Vision": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "Llama-3.2-90B-Vision": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Llama-3.1-Nemotron-70B": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "Qwen-2-72B": "Qwen/Qwen2-72B-Instruct",
    "Qwen-2-VL-72B": "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen-2.5-7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen-2.5-72B": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen-2.5-Coder-32B": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen-QwQ-32B": "Qwen/QwQ-32B-Preview",
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
    "Mistral-7B-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mistral-7B-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Mixtral-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Mixtral-8x22B": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "WizardLM2-8x22B": "microsoft/WizardLM-2-8x22B",
    "Gemma-2B": "google/gemma-2b-it",
    "Gemma-2-9B": "google/gemma-2-9b-it",
    "Gemma-2-27B": "google/gemma-2-27b-it",
    "DBRX-Instruct": "databricks/dbrx-instruct",
    "DeepSeek-LLM-67B": "deepseek-ai/deepseek-llm-67b-chat",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    "MythoMax-L2-13B": "Gryphe/MythoMax-L2-13b",
    "SOLAR-10.7B": "upstage/SOLAR-10.7B-Instruct-v1.0",
}

# Model categorization for token lengths and problematic models
MODEL_CONFIGS = {
    # Anthropic models
    "Claude-3.5-Sonnet": {"tokens": 1, "size": 200, "vendor": "anthropic"},
    "Claude-3.5-Haiku": {"tokens": 1, "size": 8, "vendor": "anthropic"},
    
    # OpenAI models
    "GPT-4-Turbo": {"tokens": 1, "size": 200, "vendor": "openai"},
    "GPT-4o": {"tokens": 1, "size": 200, "vendor": "openai"},
    "GPT-4o-Mini": {"tokens": 1, "size": 8, "vendor": "openai"},
    
    # Gemini models
    "Gemini-1.5-Flash-8B": {"tokens": 1, "size": 8, "vendor": "gemini"},
    "Gemini-1.5-Flash": {"tokens": 1, "size": 20, "vendor": "gemini"},
    "Gemini-1.5-Pro": {"tokens": 1, "size": 200, "vendor": "gemini"},
    "Gemini-1.0-Pro": {"tokens": 1, "size": 200, "vendor": "gemini"},
    "Gemini-2.0-Flash-Exp": {"tokens": 1, "size": 20, "vendor": "gemini"},
    

    # Together models (existing)
    "DeepSeek-LLM-67B": {"tokens": 1, "size": 67},
    "Mixtral-8x7B": {"tokens": 1, "size": 56},  # 8*7B
    "Mixtral-8x22B": {"tokens": 1, "size": 176},  # 8*22B
    "Llama-3-8B": {"tokens": 1, "size": 8},
    "Llama-3.2-3B": {"tokens": 1, "size": 3},
    "Llama-3.2-11B-Vision": {"tokens": 1, "size": 11},
    "Llama-3.2-90B-Vision": {"tokens": 1, "size": 90},
    "Llama-3.1-8B": {"tokens": 5, "size": 8},
    "Llama-3.1-70B": {"tokens": 1, "size": 70},
    "Llama-3.1-405B": {"tokens": 1, "size": 405},
    "Llama-3-70B": {"tokens": 1, "size": 70},
    "Qwen-2-72B": {"tokens": 1, "size": 72},
    "Qwen-2-VL-72B": {"tokens": 1, "size": 72},
    "Qwen-2.5-7B": {"tokens": 1, "size": 7},
    "Qwen-2.5-72B": {"tokens": 1, "size": 72},
    "Qwen-2.5-Coder-32B": {"tokens": 1, "size": 32},
    "Qwen-QwQ-32B": {"tokens": 1, "size": 32},
    "Mistral-7B-v0.2": {"tokens": 1, "size": 7},
    "Mistral-7B-v0.3": {"tokens": 1, "size": 7},
    
    # Models that need 5 tokens
    "Mistral-7B-v0.1": {"tokens": 5, "size": 7},
    
    "SOLAR-10.7B": {"tokens": 5, "size": 10.7},
    "Llama-3.3-70B": {"tokens": 5, "size": 70},
    "DeepSeek-V3": {"tokens": 5, "size": 67},
    "Gemma-2-9B": {"tokens": 5, "size": 9},
    "Gemma-2-27B": {"tokens": 5, "size": 27},
    "Llama-3.1-Nemotron-70B": {"tokens": 5, "size": 70},
    
    # Problematic models
    "Gemma-2B": {"error": "Inconsistent response format, many errors", "size": 2},
    "Llama-2-13B": {"error": "Inconsistent response format, often starts with 'Based on'", "size": 13},
    "DBRX-Instruct": {"error": "Does not follow the prompt well.", "size": 70},
    "MythoMax-L2-13B": {"error": "Inconsistent response format.", "size": 13},
    "WizardLM2-8x22B": {"error": "Inconsistent response format.", "size": 176},  # 8*22B
}

# Rate limiting for Anthropic (50 requests per minute)
class AnthropicRateLimiter:
    def __init__(self, max_requests=1000, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def wait_if_needed(self):
        now = time.time()
        # Remove requests older than time window
        self.requests = [t for t in self.requests if now - t < self.time_window]
        
        if len(self.requests) >= self.max_requests:
            # Wait until oldest request expires
            sleep_time = self.time_window - (now - self.requests[0])
            if sleep_time > 0:
                print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            self.requests = self.requests[1:]
        
        self.requests.append(now)

# Rate limiting for experimental Gemini models (10 requests per minute)
class GeminiRateLimiter:
    def __init__(self, model_name):
        self.is_exp = "exp" in model_name.lower()
        self.max_requests = 9 if self.is_exp else float('inf')
        if self.is_exp:
            print(f"Experimental Gemini model detected: {model_name}; max requests per minute: {self.max_requests}")
        self.time_window = 60
        self.requests = []
    
    def wait_if_needed(self):
        if not self.is_exp:
            return  # No rate limiting for non-experimental models
            
        now = time.time()
        # Remove requests older than time window
        self.requests = [t for t in self.requests if now - t < self.time_window]
        
        if len(self.requests) >= self.max_requests:
            # Wait until oldest request expires
            sleep_time = self.time_window - (now - self.requests[0])
            if sleep_time > 0:
                print(f"Rate limit reached for experimental model. Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            self.requests = self.requests[1:]
        
        self.requests.append(now)

def init_client(vendor, api_key):
    """Initialize API client based on vendor."""
    if vendor == "together":
        return Together(api_key=api_key)
    elif vendor == "openai":
        return OpenAI(api_key=api_key)
    elif vendor == "anthropic":
        return Anthropic(api_key=api_key), AnthropicRateLimiter()
    elif vendor == "gemini":
        genai.configure(api_key=api_key)
        return genai, None  # Return the genai module itself and None for rate limiter
    else:
        raise ValueError(f"Unknown vendor: {vendor}")

def chat_completion_together(client, model, request, max_tokens):
    """Make chat completion request using Together API."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request}],
            max_tokens=max_tokens,
            temperature=0,
            stream=False,
        )
        return response
    except Exception as e:
        print(f"Together API error for model {model}: {str(e)}")
        return "API-ERROR"

def chat_completion_openai(client, model, request, max_tokens):
    """Make chat completion request using OpenAI API."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request}],
            max_tokens=max_tokens,
            temperature=0,
            stream=False,
        )
        return response
    except Exception as e:
        print(f"OpenAI API error for model {model}: {str(e)}")
        return "API-ERROR"

def chat_completion_anthropic(client, rate_limiter, model, request, max_tokens):
    """Make chat completion request using Anthropic API."""
    try:
        rate_limiter.wait_if_needed()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": request}]
        )
        assert len(response.content) > 0
        return response
    except Exception as e:
        print(f"Anthropic API error for model {model}: {str(e)}")
        return "API-ERROR"

def chat_completion_gemini(client, rate_limiter, model, request, max_tokens):
    """Make chat completion request using Gemini API."""
    try:
        # Create rate limiter for this specific model if not exists
        if rate_limiter is None:
            rate_limiter = GeminiRateLimiter(model)
        
        rate_limiter.wait_if_needed()
        
        # Create model instance with specific model name
        generation_config = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 40,
            "max_output_tokens": 1,
            "response_mime_type": "text/plain",
        }
        model_instance = client.GenerativeModel(model_name=model, generation_config=generation_config)
        response = model_instance.generate_content(request)
        assert response.text is not None
        return response
    except Exception as e:
        print(f"Gemini API error for model {model}: {str(e)}")
        return "API-ERROR"

def chat_completion(client, model, request):
    """Make chat completion request with model-specific token length and vendor."""
    # Get model name and vendor
    model_name = None
    vendor = None
    api_string = None
    
    # Check in each vendor's model mappings
    if model in together_api_model_strings:
        model_name = model
        vendor = "together"
        api_string = together_api_model_strings[model]
    elif model in openai_model_strings:
        model_name = model
        vendor = "openai"
        api_string = openai_model_strings[model]
    elif model in anthropic_model_strings:
        model_name = model
        vendor = "anthropic"
        api_string = anthropic_model_strings[model]
    elif model in gemini_model_strings:
        model_name = model
        vendor = "gemini"
        api_string = gemini_model_strings[model]
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model}. Available models: {list(MODEL_CONFIGS.keys())}")
    
    max_tokens = MODEL_CONFIGS[model_name]["tokens"]
    
    if vendor == "together":
        return chat_completion_together(client, api_string, request, max_tokens)
    elif vendor == "openai":
        return chat_completion_openai(client, api_string, request, max_tokens)
    elif vendor == "anthropic":
        client, rate_limiter = client  # Unpack tuple for Anthropic
        return chat_completion_anthropic(client, rate_limiter, api_string, request, max_tokens)
    elif vendor == "gemini":
        client, rate_limiter = client  # Unpack tuple for Gemini
        return chat_completion_gemini(client, rate_limiter, api_string, request, max_tokens)
    else:
        raise ValueError(f"Unknown vendor: {vendor}")

def convert_to_chat_format(text):
    """Convert text with <extra_id_1> markers to HuggingFace chat format."""
    if "<extra_id_1>" not in text:
        return [{"role": "user", "content": text.rstrip('\n')}]
    
    turns = text.split("<extra_id_1>")
    conversation = []
    conversation.append({
        "role": "user",
        "content": turns[0].rstrip('\n')
    })
    
    for i in range(1, len(turns)):
        parts = turns[i].split("\n", 1)
        role = parts[0]
        content = parts[1].rstrip('\n')
        conversation.append({
            "role": "assistant" if role == "Assistant" else "user",
            "content": content
        })
    
    return conversation

def format_conversation(context, response_1, response_2):
    """Format the conversation using the template."""
    conversation = ""
    for message in context:
        prefix = "\n<|USER|>\n" if message["role"] == "user" else "\n<|ASSISTANT|>\n"
        conversation += f"{prefix}{message['content']}"
    
    formatted = TEMPLATE.format(
        conversation=conversation.strip(),
        response_1=response_1,
        response_2=response_2
    )
    
    return formatted

def get_preference_from_response(response):
    """Extract preference (A or B) from model response."""
    if response == "API-ERROR":
        return "API-ERROR"
    
    if isinstance(response, anthropic.types.Message):
        text = response.content[0].text.strip().lower()
    elif hasattr(response, 'text'):  # Gemini response
        text = response.text.strip().lower()
    else:
        text = response.choices[0].message.content.strip().lower()
        
    # Skip empty responses
    if not text:
        return "FORMAT-ERROR"
    
    # Strip special characters from the start
    text = text.lstrip('*_ ')
    text = text.lstrip('*_ ')
    
    # Check for various patterns
    if text == "a" or text.startswith("a is") or text.startswith("a "):
        return "A"
    if text == "b" or text.startswith("b is") or text.startswith("b "):
        return "B"
    
    # If no clear pattern, return FORMAT-ERROR
    return "FORMAT-ERROR"

def validate_models(model_names, vendor):
    """Validate models and return list of valid models, printing warnings for problematic ones."""
    valid_models = []
    for model in model_names:
        if model not in MODEL_CONFIGS:
            print(f"Warning: Unknown model {model}, skipping. Available models: {list(MODEL_CONFIGS.keys())}")
            continue
        if "error" in MODEL_CONFIGS[model]:
            print(f"Warning: Skipping model {model} due to known issues: {MODEL_CONFIGS[model]['error']}")
            continue
        if MODEL_CONFIGS[model].get("vendor", "together") != vendor:
            print(f"Warning: Skipping model {model} as it belongs to vendor {MODEL_CONFIGS[model].get('vendor', 'together')}, not {vendor}")
            continue
        valid_models.append(model)
    return valid_models

def collect_preferences(model_names, vendor, api_key, output_folder, split="val", num_examples=None):
    """Collect preferences for given models and save results."""
    # Validate models and get list of valid ones
    valid_models = validate_models(model_names, vendor)
    if not valid_models:
        print("No valid models to process. Exiting.")
        return
    
    # Load dataset
    ds = load_dataset("nvidia/HelpSteer2", data_dir="preference")['train']
    ds = ds.filter(lambda x: x['split'] == split)
    
    if num_examples is not None:
        ds = ds.select(range(min(num_examples, len(ds))))
    
    # Initialize client based on vendor
    client = init_client(vendor, api_key)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for model_name in valid_models:
        print(f"Processing model: {model_name}")
        
        # Check if output file exists and load existing results
        output_path = os.path.join(output_folder, f"{model_name}-helpsteer2-{split}.csv")
        existing_df = None
        processed_indices = set()
        
        if os.path.exists(output_path):
            print(f"Found existing results at {output_path}")
            existing_df = pd.read_csv(output_path)
            processed_indices = set(existing_df['index'].values)
            print(f"Already processed {len(processed_indices)} examples")
        
        results = []
        for idx in tqdm(range(len(ds))):
            # Skip if this example was already processed
            if idx in processed_indices:
                # print(f"Skipping index {idx} as it was already processed")
                continue
                
            item = ds[idx]
            
            # Prepare both original and swapped order requests
            context = convert_to_chat_format(item['prompt'])
            orig_request = format_conversation(context, item['response_1'], item['response_2'])
            swapped_request = format_conversation(context, item['response_2'], item['response_1'])
            
            # Get responses for both orders
            response_orig = chat_completion(client, model_name, orig_request)
            response_swapped = chat_completion(client, model_name, swapped_request)
            
            # Get raw responses and handle API errors
            if response_orig == "API-ERROR" or response_swapped == "API-ERROR":
                orig_text = "API-ERROR"
                swapped_text = "API-ERROR"
            else:
                if isinstance(response_orig, anthropic.types.Message):
                    orig_text = response_orig.content[0].text.strip()
                    swapped_text = response_swapped.content[0].text.strip()
                elif hasattr(response_orig, 'text'):  # Gemini response
                    orig_text = response_orig.text.strip()
                    swapped_text = response_swapped.text.strip()
                else:  # Together/OpenAI response
                    orig_text = response_orig.choices[0].message.content.strip()
                    swapped_text = response_swapped.choices[0].message.content.strip()
            
            pref_orig = get_preference_from_response(response_orig)
            pref_swapped = get_preference_from_response(response_swapped)
            
            # Calculate final preference
            final_pref = 0  # Default no preference
            if pref_orig == "B" and pref_swapped == "A":
                final_pref = 1  # Consistently prefers second response
            elif pref_orig == "A" and pref_swapped == "B":
                final_pref = -1  # Consistently prefers first response
            
            results.append({
                'split': split,
                'index': idx,
                'model': model_name,
                'preference_orig_order': pref_orig,
                'preference_swapped_order': pref_swapped,
                'preference': final_pref,
                'response_orig': orig_text,
                'response_swapped': swapped_text
            })
        
        # Combine new results with existing ones if any
        if existing_df is not None and not existing_df.empty:
            new_df = pd.DataFrame(results)
            df = pd.concat([existing_df, new_df], ignore_index=True)
            # Sort by index to maintain order
            df = df.sort_values('index').reset_index(drop=True)
        else:
            df = pd.DataFrame(results)
        
        # Save combined results
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path} (total {len(df)} examples)")

def parse_args():
    parser = argparse.ArgumentParser(description='Collect model preferences on HelpSteer2 dataset')
    parser.add_argument('--api_key', type=str, 
                      help='API key for the selected vendor')
    parser.add_argument('--vendor', type=str, default='together',
                      choices=['together', 'openai', 'anthropic', 'gemini'],
                      help='API vendor to use')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--models', nargs='+',
                      help='List of model names to evaluate')
    group.add_argument('--run_all', action='store_true',
                      help='Run all non-problematic models')
    parser.add_argument('--output_folder', type=str, default='results',
                      help='Folder to save results')
    parser.add_argument('--split', type=str, default='train',
                      choices=['train', 'val'],
                      help='Dataset split to use')
    parser.add_argument('--num_examples', type=int, default=None,
                      help='Number of examples to process (default: all)')
    parser.add_argument('--max_size', type=float, default=None,
                      help='Only run models smaller than this size in billions (e.g., 10 for models under 10B)')
    
    args = parser.parse_args()
    
    # Set API key from environment variable if not provided
    if args.api_key is None:
        env_var = {
            'together': 'TOGETHER_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'gemini': 'GEMINI_API_KEY'
        }[args.vendor]
        args.api_key = os.environ.get(env_var)
        if args.api_key is None:
            raise ValueError(f"API key must be provided either via --api_key or {env_var} environment variable")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    if args.run_all:
        # Get all non-problematic models under max_size (if specified) for the selected vendor
        models_to_run = [model for model in MODEL_CONFIGS.keys() 
                        if "error" not in MODEL_CONFIGS[model]
                        and MODEL_CONFIGS[model].get("vendor", "together") == args.vendor
                        and (args.max_size is None or MODEL_CONFIGS[model]["size"] < args.max_size)]
        size_msg = f" under {args.max_size}B" if args.max_size is not None else ""
        print(f"Running all non-problematic {args.vendor} models{size_msg} ({len(models_to_run)} models):")
        for model in sorted(models_to_run):
            print(f"  - {model} ({MODEL_CONFIGS[model]['size']}B)")
    else:
        models_to_run = args.models
        # Validate model names and size if specified
        for model in models_to_run:
            if model not in MODEL_CONFIGS:
                raise ValueError(f"Unknown model: {model}. Available models: {list(MODEL_CONFIGS.keys())}")
            if MODEL_CONFIGS[model].get("vendor", "together") != args.vendor:
                raise ValueError(f"Model {model} belongs to vendor {MODEL_CONFIGS[model].get('vendor', 'together')}, not {args.vendor}")
            if args.max_size is not None and MODEL_CONFIGS[model]["size"] >= args.max_size:
                raise ValueError(f"Model {model} is {MODEL_CONFIGS[model]['size']}B, which exceeds max size of {args.max_size}B")
    
    collect_preferences(
        model_names=models_to_run,
        vendor=args.vendor,
        api_key=args.api_key,
        output_folder=args.output_folder,
        split=args.split,
        num_examples=args.num_examples
    )
