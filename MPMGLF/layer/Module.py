import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from layer import Graph_Generate,gcn,Embedding,TimesBlock
from torch.optim import Optimizer
import sys





class Module(nn.Module):
    def __init__(self,config):
        super(Module, self).__init__()
        if not config.no_cuda and torch.cuda.is_available():
            # print("[using CUDA]")
            self.device = torch.device("cuda" if config.cuda_id < 0 else  'cuda:%d' % config.cuda_id)
            cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        self.config = config
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.layer = config.e_layers
        self.input_graph_knn_size = config.input_graph_knn_size

       
        self.embedding = Embedding.DataEmbedding(c_in = config.c_in,d_model = config.d_model,embed_type = config.embed,freq = config.freq,dropout = config.dropout)
        self.TimeBlocks = nn.ModuleList([TimesBlock.TimesBlock(configs = config) for i in range(config.e_layers)])
        self.GenerateGraph = Graph_Generate.GraphStructureLearning(config)
        self.gcns = nn.ModuleList([gcn.GCN(config.d_model, config.nhid, config.outfeature, config.graph_hops, config.graph_drop, config.graph_batchnorm) for i in range(config.e_layers)])
        self.layer_norm = nn.LayerNorm(config.outfeature)
        self.projection = nn.Linear(config.d_model, config.c_out, bias=True)
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)

    def prepare_init_fraph(self,x):
        x = x.transpose(1, 2)
        X_norm = X / (torch.norm(X, p=2, dim=-1, keepdim=True) + 1e-8)
        cosine_similarity = torch.bmm(X_norm, X_norm.transpose(1, 2))
        adj = (cosine_similarity >= omega).float()
        return adj


    def batch_normalize_adj(self,mx, mask=None):
   
        rowsum = torch.clamp(mx.sum(1), min=1e-5) 
        r_inv_sqrt = torch.pow(rowsum, -0.5) 
        if mask is not None:
            r_inv_sqrt = r_inv_sqrt * mask

        r_mat_inv_sqrt = []
        for i in range(r_inv_sqrt.size(0)):
            r_mat_inv_sqrt.append(torch.diag(r_inv_sqrt[i]))
        r_mat_inv_sqrt = torch.stack(r_mat_inv_sqrt, 0) 
        return torch.matmul(torch.matmul(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

    def to_cuda(self,x, device=None):
        if device:
            x = x.to(device)
        return x

    def get_adj(self,x, initknn): 
      
        graphadj = self.GenerateGraph(x)
         adj = self.config.graph_skip_conn * initknn + (1 - self.config.graph_skip_conn) * graphadj

        return graphadj,adj

    def forecast(self,x,adj):
      
        xtime = x
        for i in range(self.layer):
            xtime = self.layer_norm(self.gcns[i](self.layer_norm(self.TimeBlocks[i](xtime)), adj))
            xtime = xtime+x
        out = self.projection(xtime)  
        return out, xtime




    def forward(self,x,adj):
        out,xc = self.forecast(x,adj)
        return out,xc
      
class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        self.args = args

        self.criterion = F.mse_loss 
        self.score_func = F.mse_loss
        self.metric_name = 'MSE'

        
        self.network = Module(args)

    
        num_params = 0
        for name,p in self.network.named_parameters():
             num_params += p.numel()
        print("# There are {} parameters in total\n".format(num_params))

          self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.learning_rate)



    def init_saved_network(self,fname):

        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        self.state_dict = saved_params['state_dict']
        self.network = Module(self.args)

      
        if self.state_dict: 
            merged_state_dict = self.network.state_dict()
            for k, v in self.state_dict.items():
                if k in merged_state_dict: 
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)  


