import torch
import torch.nn as nn
import math
from models.PerceiverCLSv4_RC import PerceiverCLS
from asteroid_filterbanks import Encoder, ParamSincFB
from models.RawNetBasicBlock import Bottle2neck, PreEmphasis
from models.FBank import Fbank

class PerceiverWrapper(nn.Module):
    def __init__(self, encoder, embedding_model,encoder_type):
        super(PerceiverWrapper, self).__init__()
        self.encoder = encoder
        self.embedding_model = embedding_model
        self.encoder_type = encoder_type

    def forward(self, x, label=None):
        if self.encoder_type == "fbank":
            x = self.encoder(x)
        elif self.encoder_type == "1dconv":
            x = x.unsqueeze(1)
            x = self.encoder(x)
            x = x.permute(0,2,1)
        else:
            raise Exception("invalid encoder type")
        x = self.embedding_model(x)
        return x

def MainModel(**kwargs):
    encoder_type = kwargs["encoder_type"]
    if encoder_type == "fbank":
        encoder = Fbank(**kwargs)
    elif encoder_type == "1dconv":
        encoder = torch.nn.Conv1d(1,kwargs["n_mels"],1)
    else:
        raise Exception("invalid encoder type")
    embedding_model = PerceiverCLS(**kwargs)
    return PerceiverWrapper(encoder, embedding_model, encoder_type)

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