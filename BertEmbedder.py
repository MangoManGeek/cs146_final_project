import torch
import torch.nn as nn
from transformers import *

EMBEDDING_SIZE = 768

class BertEmbedder(nn.Module):
    def __init__(self):
        super(BertEmbedder, self).__init__()
        self.hidden_size = 768
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        outputs = model(**inputs)
        return outputs.last_hidden_state