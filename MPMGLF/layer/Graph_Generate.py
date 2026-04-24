import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from layer.SelfAttention import MultiHeadSelfAttention


class GraphStructureLearning(nn.Module):
    def __init__(self, config):
        super(GraphStructureLearning, self).__init__()

        if not config.no_cuda and torch.cuda.is_available():
            print("[using CUDA]")
            self.device = torch.device("cuda" if config.cuda_id < 0 else f'cuda:{config.cuda_id}')
            cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

        self.G = config.G
        self.K = config.K
        self.layer_num = config.MLPNums

        assert isinstance(config.ggdmodel, list), 'ggdmodel parameter must be a list'
        self.hidden_sizes = config.ggdmodel
        self.NumAttentionHeads = config.NumAttentionHeads
        self.HiddenDropoutProb = config.HiddenDropoutProb
        self.HiddenSize = config.HiddenSize

        self.PointNum = config.seq_len + config.pred_len
        self.feature_num = config.d_model

        self.MLPs = nn.ModuleList()
        for num in range(self.layer_num):
            mlp_layer = nn.Linear(self.PointNum, self.hidden_sizes[num])
            self.MLPs.append(mlp_layer)

        self.MHSA = MultiHeadSelfAttention(
            self.NumAttentionHeads, 
            (config.seq_len + config.pred_len), 
            self.HiddenSize,
            self.HiddenDropoutProb
        ).to(self.device)

    def forward(self, x):
        x = x.transpose(1, 2)
        x_row_sum = torch.sum(x, 2)
        x_p2 = torch.sqrt(torch.sum(torch.pow(x, 2), 2))
        m = torch.div(x_row_sum, x_p2)

        simij = torch.matmul(x, x.transpose(1, 2)) / torch.matmul(
            torch.norm(x, p=2, dim=-1, keepdim=True),
            torch.norm(x, p=2, dim=-1, keepdim=True).transpose(1, 2)
        )

        prox = [x]
        prom = [m]
        prosimij = [simij]

        for MLP in self.MLPs:
            hidden_v = MLP(x)
            prox.append(hidden_v)
            hidden_v_row_sum = torch.sum(hidden_v, 2)
            hidden_v_p2 = torch.sqrt(torch.sum(torch.pow(hidden_v, 2), 2))
            prox_m = torch.div(hidden_v_row_sum, hidden_v_p2)
            prom.append(prox_m)

            p_simij = torch.matmul(hidden_v, hidden_v.transpose(1, 2)) / torch.matmul(
                torch.norm(hidden_v, p=2, dim=-1, keepdim=True),
                torch.norm(hidden_v, p=2, dim=-1, keepdim=True).transpose(1, 2)
            )
            prosimij.append(p_simij)

        F_list = []
        assert len(prox) == len(prom), "Feature and mass count mismatch"
        assert len(prosimij) == len(prom), "Distance and mass count mismatch"
        assert len(prox) == len(prosimij), "Feature and distance count mismatch"

        for ith in range(len(prom)):
            F_matrix = torch.zeros((x.shape[0], x.shape[1], x.shape[1]), device=self.device)
            for batch_num in range(x.shape[0]):
                for m_idx in range(prom[ith].shape[1]):
                    for n_idx in range(prom[ith].shape[1]):
                        F_ij = self.G * prom[ith][batch_num][m_idx] * prom[ith][batch_num][n_idx] * \
                               torch.pow(prosimij[ith][batch_num][m_idx][n_idx], 2)
                        F_matrix[batch_num][m_idx][n_idx] = F_ij
            F_list.append(F_matrix)

        A_ij = []
        for F_th in F_list:
            for batch_num in range(x.shape[0]):
                value, _ = torch.topk(F_th[batch_num], self.K, 1)
                for i in range(len(F_th[batch_num])):
                    for j in range(len(F_th[batch_num][i])):
                        if F_th[batch_num][i][j] not in value[i]:
                            F_th[batch_num][i][j] = 0
            A_ij.append(F_th)

        A = []
        relu = nn.ReLU(True)
        for i in range(len(A_ij)):
            A_ij[i] = relu(A_ij[i])
            A_symi = []
            for j in range(len(A_ij[i])):
                A_symi.append((A_ij[i][j] + A_ij[i][j].T) / 2)
            A.append(torch.stack(A_symi).to(self.device))

        GravA = torch.stack(A).to(self.device)
        GravA_end = []
        GravA = GravA.transpose(1, 0)

        for batch_num in range(GravA.shape[0]):
            GravA_batch = GravA[batch_num].view(len(GravA[batch_num]), -1).to(self.device)
            GravA_end_batch = self.MHSA(GravA_batch).to(self.device)
            GravA_end.append(GravA_end_batch)

        GravA_end = torch.stack(GravA_end).to(self.device)
        GravA_end = torch.where(GravA_end > 0.5, GravA_end, torch.tensor(0.0, device=self.device))
        return GravA_end


