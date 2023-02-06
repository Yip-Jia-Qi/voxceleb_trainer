import torch
import torch.nn as nn
import math
from PerceiverCLSv4_RC import PerceiverCLS
from FilterBank import Filterbank

class PerceiverWrapper(nn.Module):
    def __init__(self, encoder, embedding_model):
        super(PerceiverWrapper, self).__init__()
        self.encoder = encoder
        self.embedding_model = embedding_model

    def forward(self, x, label=None):
        x = self.encoder(x)
        x = self.embedding_model(x)
        return x

def MainModel(**kwargs):
    encoder = Filterbank(n_mels = kwargs['ch_in'])
    embedding_model = PerceiverCLS(**kwargs)
    return PerceiverWrapper(encoder, embedding_model)

if __name__ == "__main__":
    testM = MainModel(
                ch_in = 80,
                latent_dim=192, 
                embed_dim=256,
                embed_reps=2,
                attn_mlp_dim=256, 
                trnfr_mlp_dim=256, 
                trnfr_heads=8, 
                dropout=0.2, 
                trnfr_layers=3, 
                n_blocks=2, 
                max_len = 10000,
                final_layer = '1dE')
    print(testM)