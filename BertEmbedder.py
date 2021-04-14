import torch
import torch.nn as nn
from transformers import *

EMBEDDING_SIZE = 768

class BertEmbedder(nn.Module):
    def __init__(self):
        super(BertEmbedder, self).__init__()
        self.hidden_size = 768
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        # inputs --> returned from tokenizer()
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
        return outputs.last_hidden_state