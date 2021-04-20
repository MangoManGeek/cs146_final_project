from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from torch import nn
import numpy as np
import torch
class Elmo_Embedding_layer(nn.Module):
    def __init__(self):
        super(Elmo_Embedding_layer, self).__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" #original
        
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" # original
        
       # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"

       # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"

        self.elmo = Elmo(options_file, weight_file, 1, dropout=0, requires_grad = True)

        self.hidden_size = 1024
        #self.hidden_size = 512
        self.vocab_size = 30522
        
    def forward(self,sentences):
        # Input sentences : batch_size, max sentence length, max word length(50)
        # Output: (batch_size, max sentence length, embedding_dim(1024)
        # return self.elmo(batch_to_ids(sentences))["elmo_representations"]
        rv = self.elmo(sentences)["elmo_representations"][0]
        # rv = torch.stack(rv)[0]
        # print(rv.shape)
        return rv
    
elmo = Elmo_Embedding_layer()
sentences = [['First', 'word', 'is', 'good', '.'], ['First', 'word', 'is', 'awesome', '.']]
sentences_comp = [['First', 'bad', 'thing', 'is', 'love'], ['First', 'bad', 'thing', 'is', 'hate']]
result = elmo.elmo(batch_to_ids(sentences))["elmo_representations"]
result_comp = elmo.elmo(batch_to_ids(sentences_comp))["elmo_representations"][0]
test = result_comp.clone().detach().sum(dim=1)
print(test)
print(test.shape)
#t = result[0].detach().numpy()
#t0, t1 = t[0].flatten(), t[1].flatten()
#t = result_comp[0].detach().numpy()
#t2, t3 = t[0].flatten(), t[1].flatten()
#ls = [t0, t1, t2, t3]
#res = np.zeros((4,4))
#for i, x in enumerate(ls):
#    for j, y in enumerate(ls):
#        res[i,j] = np.sqrt(np.sum((x-y)**2))
#print(res)

# should be fine here

    
