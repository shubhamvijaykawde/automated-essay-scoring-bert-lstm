import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name="bert-base-uncased"):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item
