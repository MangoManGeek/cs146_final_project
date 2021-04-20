
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from BertEmbedder import *
from Elmo_embedding import *


class Transformer_Classifier(nn.Module):
    def __init__(self, embedder_type, window_size, num_classes, nhead=8, num_layers=3):
        """
        The Model class implements the LSTM-LM model.
        Feel free to initialize any variables that you find necessary in the
        constructor.
        :param vocab_size: The number of unique tokens in the data
        :param rnn_size: The size of hidden cells in LSTM/GRU
        :param embedding_size: The dimension of embedding space
        """
        super().__init__()
        # TODO: initialize the vocab_size, rnn_size, embedding_size
        # self.vocab_size = vocab_size
        # self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout_rate = 0.3
        self.linear1_out_size = 128

        # TODO: initialize embeddings, LSTM, and linear layers
        if embedder_type == 'bert':
            self.embeddings = BertEmbedder()
        else:
            self.embeddings = Elmo_Embedding_layer()
        self.num_classes = num_classes
        self.embedding_size = self.embeddings.hidden_size
        self.hidden_size = self.embedding_size
        self.window_size = window_size
        self.pos_encoder = Position_Encoding_Layer(self.window_size, self.embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = torch.nn.Linear(in_features = self.hidden_size*self.window_size,
                                    out_features = self.linear1_out_size)
        self.output_layer = torch.nn.Linear(in_features = self.linear1_out_size,
                                    out_features = self.num_classes)
        self.gelu = torch.nn.GELU()

        #self.hidden_in = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        #self.cell_in = torch.randn(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs):

        """
        Runs the forward pass of the model.
        :param inputs: word texts (tokens) of shape (batch_size, window_size)
        :param lengths: array of actual lengths (no padding) of each input
        :return: the logits, a tensor of shape
                 (batch_size, window_size, vocab_size)
        """
        # TODO: write forward propagation

        # make sure you use pack_padded_sequence and pad_padded_sequence to
        # reduce calculation

        embedding_output = self.embeddings(inputs) # batch * window * embeddings
        positional_embeddings = self.pos_encoder(embedding_output)
#        print(positional_embeddings.shape,"@@@@")
#        transpose_tensor = torch.transpose(positional_embeddings, 0, 1)
#        print(transpose_tensor.shape,"!!!!")
        encoder_output = self.transformer_encoder(torch.transpose(positional_embeddings, 0, 1))
        concated_tensor = torch.reshape(torch.transpose(encoder_output, 0, 1), (positional_embeddings.shape[0], -1))
        linear_out = self.linear(concated_tensor)
        linear_out = self.gelu(linear_out)
        logist = self.output_layer(linear_out)
        return logist
    
class Position_Encoding_Layer(nn.Module):
    def __init__(self, window_sz, emb_sz):
        super(Position_Encoding_Layer, self).__init__()
#        self.positional_embeddings = nn.parameter.Parameter(torch.rand([window_sz, emb_sz]))
        self.positional_embeddings = nn.parameter.Parameter(
            torch.nn.init.xavier_uniform_(
            torch.tensor(torch.rand([window_sz, emb_sz]))))
        
        
    def forward(self,x):
        return x+self.positional_embeddings