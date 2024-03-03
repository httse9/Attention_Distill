import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from os import path
import os
import numpy as np

class LinearLayer(nn.Module):

    def __init__(self, sequence_len, embedding_dim, path=None):
        super().__init__()
        self.path = path
        os.makedirs(path, exist_ok=False)

        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        self.net = nn.Linear(sequence_len * embedding_dim, sequence_len * embedding_dim)

    def forward(self, sequence):
        """
        sequence: sequence of token embeddings, size (batch size, sequence length, embedding_dim)

        return: same size as sequence
        """
        bs, sequence_length, embedding_dim = sequence.size()
        flattened_sequence = sequence.view(bs, -1)      # (batch size, sequence length * embedding dim)

        return self.net(flattened_sequence).view(bs, sequence_length, embedding_dim)
    
    def visualize(self):
        """
        visualize the paramters of this layer
        """
        assert self.path is not None
        weight_matrix = self.net.weight.detach().cpu()
        size_ratio = (self.sequence_len / 12) * (self.embedding_dim / 128)
        plt.figure(figsize=(6.4 * size_ratio, 4.8 * size_ratio))
        plt.imshow(weight_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar()

        # set grids for better visualization
        ax = plt.gca()
        ax.set_xticks(np.arange(0, self.sequence_len * self.embedding_dim, self.embedding_dim))
        ax.set_yticks(np.arange(0, self.sequence_len * self.embedding_dim, self.embedding_dim))
        ax.grid(color='w')

        plt.savefig(path.join(self.path, "weight.png"), dpi=300)
        plt.clf()
        

    def save_model(self):
        assert self.path is not None
        torch.save(self.state_dict(), path.join(self.path, "model.pt"))

    @staticmethod
    def load_model(path):
        pass

