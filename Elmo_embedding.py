from allennlp.modules.elmo import Elmo, batch_to_ids
from torch import nn
import torch
class Elmo_Embedding_layer(nn.Module):
    def __init__(self):
        super(Elmo_Embedding_layer, self).__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"  
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
        # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0, requires_grad = False)

        self.hidden_size = 1024
        # self.hidden_size = 256

        self.vocab_size = 30522
        
    def forward(self,sentences):
        # Input sentences : batch_size, max sentence length, max word length(50)
        # Output: (batch_size, max sentence length, embedding_dim(1024)
        # return self.elmo(batch_to_ids(sentences))["elmo_representations"]
        rv = self.elmo(sentences)["elmo_representations"][0]
        # rv = torch.stack(rv)[0]
        # print(rv.shape)
        return rv
    
#elmo = Elmo_Embedding_layer()
#sentences = [['First', 'sentence', 'is', 'over', '.'], ['Another', 'test','.']]
#print(elmo(sentences))