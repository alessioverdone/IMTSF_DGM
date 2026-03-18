import torch
import torch.nn as nn
from einops import rearrange
from torch_geometric.nn import GCN

from src.layers.mds import DGMmodule
from src.layers.utils import build_graph_from_mask

import torch
import torch.nn as nn

class QueryPooling(nn.Module):
    def __init__(self, N_2, F, num_heads=4):
        super().__init__()
        self.N_2 = N_2
        # N_2 query learnable
        self.queries = nn.Parameter(torch.randn(1, N_2, F))
        self.attn = nn.MultiheadAttention(F, num_heads, batch_first=True)

    def forward(self, x, batch):
        # x: (B*N_i, F) con N_i variabile
        B = batch.max().item() + 1

        # padding: porta ogni grafo a (B, N_max, F) con key_padding_mask
        counts = batch.bincount(minlength=B)          # (B,)
        N_max = counts.max().item()

        x_padded = torch.zeros(B, N_max, x.size(-1), device=x.device)
        mask = torch.ones(B, N_max, dtype=torch.bool, device=x.device)  # True = ignora

        idx = 0
        for b in range(B):
            n = counts[b].item()
            x_padded[b, :n] = x[idx:idx+n]
            mask[b, :n] = False  # questi sono validi
            idx += n

        # cross-attention: query learnable, key/value dai nodi
        q = self.queries.expand(B, -1, -1)            # (B, N_2, F)
        out, _ = self.attn(q, x_padded, x_padded, key_padding_mask=mask)
        # out: (B, N_2, F)

        return out.reshape(B * self.N_2, x.size(-1))  # (B*N_2, F)


class Prova(nn.Module):
    def __init__(self, args):
        super(Prova, self).__init__()
        self.args = args
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.batch_size = None
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)
        self.nodevec = nn.Embedding(self.N, d_model)
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )
        self.dgm = DGMmodule(args)
        self.gcn = GCN(in_channels=d_model,
                       out_channels=d_model,
                       hidden_channels=d_model,
                       num_layers=2)
        self.pool = QueryPooling(args.ndim, d_model, num_heads=4)

    def LearnableTE(self, tt):
        # learnable continuous time embeddings
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def forward(self, time_steps_to_predict, X, truth_time_steps, mask=None):
        out_dict = dict()
        # Preprocess (patch input to normal input)
        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.squeeze().permute(0, 2, 1).unsqueeze(-1)  # levo M, permuto e aggiungo dimensione per espandere h_dim, [B, N, T_in, 1]
        truth_time_steps = truth_time_steps.squeeze().permute(0, 2, 1).unsqueeze(-1)  # [B, N, T_in, 1]
        mask = mask.squeeze().permute(0, 2, 1).unsqueeze(-1)  # [B, N, T_in, 1]

        # Encoder
        X = self.obs_enc(X)  # [B, N, T_in, 1] -> observation encoder -> [B, N, T_in, D]
        te_his = self.LearnableTE(truth_time_steps)  # [B, N, T_in, 1] -> time-step encoder -> [B, N, T_in, D]
        var_emb = self.nodevec.weight.view(1, N, 1, self.args.hid_dim).repeat(B, 1, L_in, 1)  # [B, N, T_in, 1] -> variable encoder -> [B, N, T_in, D]
        X = self.relu(X + var_emb + te_his)  # node-graph embedding  -> [B, N, T_in, D]

        # GNN
        graph = build_graph_from_mask(X, mask.squeeze())  # Data object: graph.x:[(mask == 1.).sum(), D], graph.edge_index:[2,E]
        h = self.gcn(x=graph.x,
                      edge_index=graph.edge_index,
                      batch=graph.batch)
        h, l_probs = self.dgm(graph.x, graph.edge_index)
        out_dict['l_probs'] = l_probs

        # DGM
        h = self.pool(h, graph.batch)  # [(mask == 1.).sum(), D] -> pool -> [B*N, D]
        h = torch.reshape(h, (B,N,-1))  # [B, N, D]

        # Decoder
        L_pred = time_steps_to_predict.shape[-1]  # 40
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # # [B, N, L_out, D]
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # [B, L_out] -> [B, N, L_out, 1]
        te_pred = self.LearnableTE(time_steps_to_predict)  # [B, N, L_out, 1] -> [B, N, L_out, D]
        h = torch.cat([h, te_pred], dim=-1)  # [B, N, L_out, D] -> [B, N, L_out, 2D]
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)  # [B, N, L_out, 2D] -> decoder -> [1, B, L_out, N]
        out_dict['pred_y'] = outputs
        return out_dict



class HipatchOur(nn.Module):
    def __init__(self, args):
        super(HipatchOur, self).__init__()
        self.args = args
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.batch_size = None
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)
        self.nodevec = nn.Embedding(self.N, d_model)
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )


    def LearnableTE(self, tt):
        # learnable continuous time embeddings
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def forward(self, time_steps_to_predict, X, truth_time_steps, mask=None):
        B, M, L_in, N = X.shape
        self.batch_size = B

        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B,N,M,T_in,1)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B,N,M,T_in,1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B,N,M,T_in,1)

        # # Encoder
        X = self.obs_enc(X)  # (8,41,1,57,64 )observation encoder (8,41,1,57,64 )
        te_his = self.LearnableTE(truth_time_steps)  # time-step encoder (8,41,1,57,64 )
        var_emb = self.nodevec.weight.view(1, N, 1, 1, self.args.hid_dim).repeat(B, 1, M, L_in, 1)  # variable encoder (8,41,1,57,64 )
        X = self.relu(X + var_emb + te_his)  # node-graph embedding (8,41,1,57,64) -> (B,C,1,T_in,H)
        #
        # h = self.IMTS_Model(X, mask, truth_time_steps)  # (8,41,64)
        #
        # (B, C, 1, T_in, H) -> model() -> (B,C,H) (then T_in=N)
        # 1) (B, C, 1, N, H)  -> (B, C, N, H) (squeeze)
        X = torch.squeeze(X)
        mask = torch.squeeze(mask)   # (B,C,N)
        # 2) (B, C, N, H) -> (B, N, C, H) (rearrange)
        # 3) (B, N, C, H) -> (B * N, C * H) (rearrange)
        # X = rearrange(X, 'b c n h -> (b n c) h')
        # 4) DGM
        graph = build_graph_from_mask(X, mask)
        h_ = self.gcn(x=graph.x,
                     edge_index=graph.edge_index,
                     batch=graph.batch,),

        h = h_[0]
        h = self.pool(h, graph.batch)
        h = torch.reshape(h, (B,N,-1))

        # """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]  # 40
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (8,41,40,64)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (8,41,40,1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (8,41,40,64)
        h = torch.cat([h, te_pred], dim=-1)  # (8,41,40,128)
        # h = torch.randn((B, N, L_pred, 2*self.args.hid_dim), requires_grad=True).to(self.args.device)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)  # (1,8,40,41)
        return outputs
