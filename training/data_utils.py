import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class UncertaintyDataset(Dataset):
    def __init__(self, jsonl_file, max_length=512):
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set pad_token_id for the model
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        with open(jsonl_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                prompt = entry['prompt']
                response = entry['response']
                
                # Format the input
                text = f"Question: {prompt}\nAnswer: {response}"
                
                # Tokenize
                tokenized = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                self.data.append({
                    'input_ids': tokenized['input_ids'].squeeze(),
                    'attention_mask': tokenized['attention_mask'].squeeze()
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'attention_mask': self.data[idx]['attention_mask']
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}