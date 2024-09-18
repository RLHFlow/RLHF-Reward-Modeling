import random
import re
import tqdm
from datasets import load_dataset
from datasets import Dataset, DatasetDict
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_dataset", type=str)
parser.add_argument("--output_dataset", type=str)
args = parser.parse_args()


def get_fields(messages) -> dict[str, str]:
  delimiters = r'\[CONTEXT\]|\[RESPONSE A\]|\[RESPONSE B\]'
  # Split the string
  result = re.split(delimiters, messages[0]['content'])
  # Remove empty strings that may result from consecutive delimiters or delimiters at the start/end
  result = [x for x in result if x]
  assert len(result) == 3
  assert messages[1]['content'] in ['A', 'B', 'Same']
  if messages[1]['content'] == 'A':
    return {
        'context': result[0],
        'response_w': result[1],
        'response_l': result[2],
        'neutral': False
    }
  elif messages[1]['content'] == 'B':
    return {
        'context': result[0],
        'response_l': result[1],
        'response_w': result[2],
        'neutral': False
    }
  else:
    return {
        'context': result[0],
        'response_w': result[1],
        'response_l': result[2],
        'neutral': True
    }


def to_messages(fields: dict[str, str]) -> str:
  context = fields['context']
  neutral = fields["neutral"]
  if random.randint(0,1):
    response_a = fields['response_w']
    response_b = fields['response_l']
    label = "A"
  else:
    response_a = fields['response_l']
    response_b = fields['response_w']
    label = "B"
  if neutral:
    label = "Same"
  message_0 = {
      "role": "user",
      "content": f"[CONTEXT]{context}" +
          f"[RESPONSE A]{response_a}" +
          f"[RESPONSE B]{response_b}"
  }
  message_1 = {
      "role": "assistant",
      "content": label
  }
  return [message_0, message_1]


def get_augmented(data):
  data_i = data
  data_j = data_i.copy()
  random.shuffle(data_j)
  data_k = data_j.copy()
  random.shuffle(data_k)
  for ex_i, ex_j, ex_k in zip(data_i, data_j, data_k):
    xi = ex_i['context']
    xj = ex_j['context']
    xk = ex_k['context']
    ywi = ex_i['response_w']
    ywj = ex_j['response_w']
    ywk = ex_k['response_w']
    yli = ex_i['response_l']
    ylj = ex_j['response_l']
    ylk = ex_k['response_l']
    # xi_ywi_ywj
    yield {
        "context": xi,
        "response_w": ywi,
        "response_l": ywj,
        "neutral": False
    }
    # xi_ywi_ywk
    yield {
        "context": xi,
        "response_w": ywi,
        "response_l": ywk,
        "neutral": False
    }
    # xi_ywi_ylj
    yield {
        "context": xi,
        "response_w": ywi,
        "response_l": ylj,
        "neutral": False
    }
    # xi_ywi_ylk
    yield {
        "context": xi,
        "response_w": ywi,
        "response_l": ylk,
        "neutral": False
    }
    # xi_yli_ywj
    yield {
        "context": xi,
        "response_w": yli,
        "response_l": ywj,
        "neutral": False
    }
    # xi_yli_ywk
    yield {
        "context": xi,
        "response_w": yli,
        "response_l": ywk,
        "neutral": False
    }
    # xi_yli_ylj
    yield {
        "context": xi,
        "response_w": yli,
        "response_l": ylj,
        "neutral": False
    }
    # xi_yli_ylk
    yield {
        "context": xi,
        "response_w": yli,
        "response_l": ylk,
        "neutral": False
    }
    # xi_ywj_ylj
    yield {
        "context": xi,
        "response_w": ywj,
        "response_l": ylj,
        "neutral": True
    }
    # xi_ywk_ylk
    yield {
        "context": xi,
        "response_w": ywk,
        "response_l": ylk,
        "neutral": True
    }
    # xi_ywj_ywk
    yield {
        "context": xi,
        "response_w": ywj,
        "response_l": ywk,
        "neutral": True
    }
    # xi_ywj_ylk
    yield {
        "context": xi,
        "response_w": ywj,
        "response_l": ylk,
        "neutral": True
    }
    # xi_ywk_ylj
    yield {
        "context": xi,
        "response_w": ywk,
        "response_l": ylj,
        "neutral": True
    }
    # xi_ylj_ylk
    yield {
        "context": xi,
        "response_w": ylj,
        "response_l": ylk,
        "neutral": True
    }


def process_data(data):
  all_fields = []
  for d in tqdm.tqdm(data):
    try:
      all_fields.append(get_fields(d['messages']))
    except:
      print(d['messages'])
  for fields in tqdm.tqdm(get_augmented(all_fields)):
    yield to_messages(fields)


ds = load_dataset(args.input_dataset, split='train')
processed_messages = list(process_data(list(ds)))
new_ds = {'train': []}
for m in processed_messages:
  new_ds['train'].append({'messages': m})
dict_data = {'messages': [x['messages'] for x in new_ds['train']]}
dataset = Dataset.from_dict(dict_data)
DatasetDict({"train": dataset}).push_to_hub(args.output_dataset)

