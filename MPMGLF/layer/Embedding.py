import torch
import torch.nn as nn
import math
import sys
class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_length = 5000):
        super(PositionEmbedding,self).__init__()
        pe = torch.zeros(max_length, d_model).float() 
        pe.requires_grad = False
       postion = torch.arange(0, max_length).float().unsqueeze(1)
        w_k = (torch.arange(0,d_model,2).float() * -(math.log(10000.0)/d_model)).exp() # 128，
       pe[:, 0::2] = torch.sin(postion * w_k)
        pe[:, 1::2] = torch.cos(postion * w_k)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) 

    def forward(self, x):
        return self.pe[:, :x.size(1)] 


class TokenEmbedding(nn.Module):
    def __init__(self, x, d_model):
        super(TokenEmbedding,self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
    self.tokenConv = nn.Conv1d(in_channels = x,out_channels = d_model,
                                   kernel_size = 3,padding = padding,padding_mode = 'circular', bias=False)

        for m in self.modules(): 
            if isinstance(m, nn.Conv1d):
                
                nn.init.kaiming_normal_(
                    m.weight, mode = 'fan_in', nonlinearity = 'leaky_relu'
                )

    def forward(self, x): 
       x = self.tokenConv(x.permute(0,2,1)).transpose(1,2) 
        return x

class  FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
  
        pe = torch.zeros(c_in, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        w_k = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * w_k)
        pe[:, 1::2] = torch.cos(position * w_k)
      self.emb = nn.Embedding(c_in, d_model) 
        self.emb.weight = nn.Parameter(pe, requires_grad = False) 

    def forward(self, x):
        return self.emb(x).detach() 


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type = 'fixed', freq = 'h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

         self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        month_x = self.month_embed(x[:, :, 0])

      return month_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, freq = 'h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {
            'h': 4, 't': 5, 's': 6,
            'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3
        }
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias = False)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self,c_in,d_model,embed_type = 'fixed',freq = 'h', dropout = 0.1):
        super(DataEmbedding,self).__init__()
        self.position_embedding = PositionEmbedding(d_model = d_model)
        self.token_embedding = TokenEmbedding(x = c_in, d_model = d_model)
        if embed_type != 'TimeFeatureEmbedding':
            self.temporal_embedding = TemporalEmbedding(d_model = d_model, embed_type = embed_type, freq = freq)
        else:
            self.temporal_embedding = TimeFeatureEmbedding(d_model = d_model, freq = freq)

        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, x_mark):
 
        if x_mark is None:
            x = self.token_embedding(x) + self.position_embedding(x)
        else:
            x = self.token_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)

        return self.dropout(x)




