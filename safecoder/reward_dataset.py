import json
import os
from random import shuffle

import torch
from torch.utils.data import Dataset

from .reward import format_reward_input


def _language_from_sample(sample: dict) -> str:
    file_name = sample.get('file_name', '')
    if '.' in file_name:
        return file_name.rsplit('.', 1)[-1]
    return sample.get('language', 'unknown')


def load_reward_examples(data_dir: str, mode: str, datasets) -> list:
    examples = []
    for dataset_name in datasets:
        path = os.path.join(data_dir, mode, f'{dataset_name}.jsonl')
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                info = {
                    'language': _language_from_sample(sample),
                    'cwe': sample.get('vul_type', ''),
                    'description': sample.get('description', ''),
                }
                before = sample.get('func_src_before')
                after = sample.get('func_src_after')
                if after:
                    examples.append((format_reward_input(after, info), 1))
                if before:
                    examples.append((format_reward_input(before, info), 0))
    return examples


class RewardDataset(Dataset):
    def __init__(self, args, tokenizer, mode: str):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.examples = load_reward_examples(args.data_dir, mode, args.datasets)
        if mode == 'train':
            shuffle(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        text, label = self.examples[item]
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.args.max_num_tokens,
            return_tensors='pt',
        )
        encoded = {key: value.squeeze(0) for key, value in encoded.items()}
        encoded['labels'] = torch.tensor(label, dtype=torch.long)
        return encoded
