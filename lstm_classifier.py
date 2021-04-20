
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch

from BertEmbedder import *
from Elmo_embedding import *


class LSTM_Classifier(nn.Module):
    def __init__(self, embedder_type, rnn_size, num_classes, num_layers = 1):
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
        self.hidden_size = rnn_size
        # self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout_rate = 0.3
        self.linear1_out_size = 128

        # TODO: initialize embeddings, LSTM, and linear layers
        if embedder_type == 'bert':
        	self.embeddings = BertEmbedder()
        else:
        	self.embeddings = Elmo_Embedding_layer()

        self.embedding_size = self.embeddings.hidden_size
        self.num_classes = num_classes

        self.lstm = torch.nn.LSTM(input_size=self.embedding_size,
                                hidden_size = self.hidden_size,
                                num_layers = self.num_layers,
                                dropout = self.dropout_rate,
                                batch_first = True)
        self.linear = torch.nn.Linear(in_features = self.hidden_size,
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

        in_embeddings = self.embeddings(inputs) # batch * window * embeddings
        # packed_embeddings = torch.nn.utils.rnn.pack_padded_sequence(input = in_embeddings, 
        #                                                             lengths = lengths, 
        #                                                             batch_first = True,
        #                                                             enforce_sorted=False)
        packed_embeddings = in_embeddings
        # print(packed_embeddings)
        lstm_out, (hn, cn) = self.lstm(input = packed_embeddings)       # lstm_out shape: batch * seq_length * embedding
        # unpacked_lstm_out, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(sequence = lstm_out, 
        #                                                                             batch_first = True,
        #                                                                             total_length=inputs.shape[1])
        # unpacked_lstm_out = lstm_out
        #print(lens_unpacked.shape, "lengths")
        # print("hn: ", hn.shape)
        linear_out = self.linear(hn[0])
        linear_out = self.gelu(linear_out)
        logits = self.output_layer(linear_out)
        return logits     # batch * window * vocab