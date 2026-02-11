import torch
import torch.nn as nn
from transformers import BertModel

class BertMultiOutputRegressor(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_outputs=6):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, num_outputs)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_embedding)
