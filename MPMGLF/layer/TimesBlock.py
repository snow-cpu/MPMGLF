import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
# sys.setrecursionlimit(6000)
class Inception_Block_v1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels = 6, init_weight = True):
        super(Inception_Block_v1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 2 * i + 1, padding = i))

        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim = -1).mean(-1)
        return res

def FFT_for_Period(x, k = 2):
    xf = torch.fft.rfft(x, dim = 1) 
    frequency_list1 = abs(xf).mean(0)
    frequency_list=frequency_list1.mean(-1)
   
    frequency_list[0] = 0
     _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.cpu().detach().numpy()
   
    period = x.shape[1] // top_list 
    return period,abs(xf).mean(-1)[:, top_list] 

class TimesBlock(nn.Module):
    def __init__(self,configs):
        super(TimesBlock,self).__init__(),
        self.seq_len = configs.seq_len 
        self.pred_len = configs.pred_len 
        self.k = configs.top_k 

        self.conv = nn.Sequential(
            Inception_Block_v1(configs.d_model,configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_v1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self,x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
      
        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                    length = (((self.seq_len + self.pred_len) // period) + 1) * period
                    padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                    out = torch.cat([x, padding], dim=1) 
            else:
                length = (self.seq_len + self.pred_len)
                out = x
             out = out.reshape(B, length // period, period ,N).permute(0,3,1,2).contiguous() 
         
            out = self.conv(out)  # 8,256,6,25

          
            out = out.permute(0,2,3,1).reshape(B,-1,N) 
            res.append(out[:, :(self.seq_len + self.pred_len), :]) 

        res = torch.stack(res,dim = -1)
      
        period_weight = F.softmax(period_weight,dim = 1) 
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1,T,N,1) 
        res = torch.sum(res * period_weight, -1) 
   
        res = res + x

        return res

