import math

import torch
import torch.nn as nn
import sys

NumAttentionHeads = 4
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,NumAttentionHeads, InputSize, HiddenSize,HiddenDropoutProb):
      
        super(MultiHeadSelfAttention,self).__init__()
        if HiddenSize % NumAttentionHeads != 0:
            raise "The hidden size (%d) is not a multiple  of the number of attention heads (%d)" % (HiddenSize,NumAttentionHeads)
        self.NumAttentionHeads = NumAttentionHeads
        self.AddtentionHeadSize = int(HiddenSize / NumAttentionHeads) 
        self.AllHeadSize = HiddenSize 

        self.query  = nn.Linear(InputSize,self.AllHeadSize) 
        self.key = nn.Linear(InputSize, self.AllHeadSize) 
        self.value = nn.Linear(InputSize, self.AllHeadSize) 

        self.AttenDrop = nn.Dropout(HiddenDropoutProb) 
        self.dense = nn.Linear(HiddenSize, InputSize) 
        self.LayerNorm = LayerNorm(InputSize, eps = 1e-12) 

    def FransposeForScores(self,x):
        newXshape = x.size()[:-1] + (self.NumAttentionHeads,self.AddtentionHeadSize)
        x = x.view(*newXshape) 
        return x.permute(1,0,2) 
    def forward(self,x):
        MixedQueryLayer = self.query(x) 
        MixedKeyLayer = self.key(x) 
        MixedValueLayer = self.value(x) 
   
        queryLayer = self.FransposeForScores(MixedQueryLayer) 
        keyLayer = self.FransposeForScores(MixedKeyLayer)  
        valueLayer = self.FransposeForScores(MixedValueLayer)  
      
        attentionScores = torch.matmul(queryLayer,keyLayer.transpose(-1,-2))
        attentionScores = attentionScores/math.sqrt(self.AddtentionHeadSize)
        attentionProbs = nn.Softmax(dim=-1)(attentionScores)
 
        contextLayer = torch.matmul(attentionProbs,valueLayer) 
        contextLayer = contextLayer.permute(1,0,2).contiguous() 
        newContextLayerShape = contextLayer.size()[:-2] + (self.AllHeadSize,)
        contextLayer = contextLayer.view(*newContextLayerShape) 
         out = self.dense(contextLayer)
   
        out = self.AttenDrop(out)
        out = self.LayerNorm(out+x)

        lens = int(math.sqrt(x.size()[-1]))
        out = torch.sum(out,dim = 0).contiguous().view(lens,lens)
        return out

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

