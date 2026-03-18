import math
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
from torch.nn import functional as F

from src.layers.gcn1d import GCN1DConv, GCN1DConv_big
from src.layers.dgm import DGM_d, MLP, Identity
from src.layers.mds import STS3M


class SeasonalExpert(nn.Module):
    """
    Cattura la stagionalità (es. 24 h o 1 settimana).
    In pratica: 1) codifica ogni timestamp in sin/cos, 2) MLP sui coefficienti.
    """
    def __init__(self, lags: int, horizon: int, num_harmonics: int = 4, hidden: int = 32):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.fc = nn.Sequential(
            nn.Linear(2 * num_harmonics, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon)
        )
        self.register_buffer(
            "omega",
            torch.arange(1, num_harmonics + 1).float() * 2 * math.pi / 24  # 24 = periodo (ore)
        )

    def forward(self, x):              # x: [B, N, T, 1]
        B, N, T, _ = x.shape
        t = torch.arange(T, device=x.device).float()  # 0 … T‑1
        # sin/cos -> [T, 2*H]
        bases = torch.cat([t[:, None] * self.omega, t[:, None] * self.omega], 1)
        bases[:, :self.num_harmonics] = torch.sin(bases[:, :self.num_harmonics])
        bases[:, self.num_harmonics:] = torch.cos(bases[:, self.num_harmonics:])
        # proietta i valori sui basi (least‑squares) → [B,N,2H]
        coeffs = torch.linalg.lstsq(bases, x.squeeze(-1).transpose(1, 2).reshape(-1, T).T).solution.T
        coeffs = coeffs.view(B, N, -1)
        return self.fc(coeffs)         # [B,N,H]


class SpikeExpert(nn.Module):
    """
    Mixa convoluzioni corte (catturano spike) + GRU.
    """
    def __init__(self, in_channels: int, hidden: int, horizon: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(32, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, horizon)

    def forward(self, x):              # x: [B,N,T,1]
        B, N, T, _ = x.shape
        x = x.view(B * N, 1, T)        # conv1d → [B·N, C, T]
        z = self.cnn(x).transpose(1, 2)  # → [B·N, T, 32]
        h, _ = self.rnn(z)             # → [B·N, T, hidden]
        out = self.proj(h[:, -1])      # ultimo time step
        return out.view(B, N, -1)      # [B,N,H]




class IsolatedNodeExpert(nn.Module):
    """
    Amplifica il segnale dei nodi poco connessi (degree bassa) con pesi learnable.
    """
    def __init__(self, num_nodes: int, hidden: int, horizon: int, edge_index, edge_weight):
        super().__init__()
        # Pre‑calcolo degree (fisso)
        deg = torch.bincount(edge_index[0], weights=edge_weight, minlength=num_nodes)
        iso_score = 1.0 / (deg + 1e-3)         # più grande → più isolato
        self.register_buffer("iso_score", iso_score)  # [N]
        self.gcn = GCNConv(1, hidden)
        self.proj = nn.Linear(hidden, horizon)

    def forward(self, x, edge_index):                     # x: [B,N,T,1]
        B, N, T, _ = x.shape
        # media temporale → [B,N,1]
        x_mean = x.mean(dim=2).squeeze(-1)
        # amplifica isolati
        x_weighted = x_mean * self.iso_score   # broadcast
        # gcn → [B,N,hidden]
        out = []
        for b in range(B):
            h = self.gcn(x_weighted[b].unsqueeze(-1), edge_index)  # edge_index condiviso
            out.append(h)
        out = torch.stack(out, dim=0)
        return self.proj(out)                 # [B,N,H]


class AttentiveGating(nn.Module):
    def __init__(self, in_channels: int, num_experts: int, k: int = 6, d_model: int = 32):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_channels, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.fc = nn.Linear(d_model, num_experts)

    def forward(self, x):              # x: [B,N,T,C]
        B, N, T, C = x.shape
        x_k = x[:, :, -self.k:].transpose(1, 2)          # [B,k,N,C]
        qkv = self.proj(x_k)                             #→[B,k,N,d]
        h,_ = self.attn(qkv, qkv, qkv)                   # self‑attn
        summary = h.mean(dim=1)                          # [B,N,d]
        α = torch.softmax(self.fc(summary), dim=-1)      # [B,N,E]
        return α





# ----------  gli expert specializzati che avevi già scritto  ----------
class SeasonalExpertv2(nn.Module):
    """Proietta la serie su basi sin/cos per catturare stagionalità (es. 24 h)."""
    def __init__(self, lags: int, horizon: int, num_harmonics: int = 4, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.fc = nn.Sequential(
            nn.Linear(2 * num_harmonics, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon)
        )
        self.register_buffer(
            "omega",
            torch.arange(1, num_harmonics + 1).float() * 2 * math.pi / 24
        )

    def forward(self, x):                                      # x: [B,N,T,1]
        B, T, N, _ = x.shape
        t = torch.arange(T, device=x.device).float()
        # [T, 2H]  → sin|cos per ogni armonica
        bases = torch.cat([t[:, None] * self.omega,
                           t[:, None] * self.omega], dim=1)
        bases[:, :self.num_harmonics] = torch.sin(bases[:, :self.num_harmonics])
        bases[:, self.num_harmonics:] = torch.cos(bases[:, self.num_harmonics:])
        # least‑squares → [B·N, 2H]
        coeffs = torch.linalg.lstsq(
            bases,  # [T, 2H]
            rearrange(x, 'b t n f -> t (b n f)')).solution.T
        coeffs = coeffs.view(B, N, -1)  # [B,N, 2H]
        return self.fc(coeffs)  # [B,N,H]


class SpikeExpertv2(nn.Module):
    """Cnn corte + GRU per catturare spike e dipendenze a breve termine."""
    def __init__(self, in_channels: int, hidden: int, horizon: int, dropout: float):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.rnn = nn.GRU(32, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, horizon)

    def forward(self, x):                                      # x: [B,N,T,1]
        # # [B,T,N,F]
        B, T, N, _ = x.shape
        z = self.cnn(rearrange(x, 'b t n f -> (b n) f t')).transpose(1, 2)  # [B·N, T, 32]
        h, _ = self.rnn(z)                                     # [B·N, T, hidden]
        out = self.proj(h[:, -1])                              # ultimo step
        return out.view(B, N, -1)                              # [B,N,H]

class IsolatedNodeExpertv2(nn.Module):
    """Pesa di più i nodi con bassa degree e propaga coi GCN."""
    def __init__(self, num_nodes: int, hidden: int, horizon: int,
                 edge_index, edge_weight=None, dropout: float = 0.1):
        super().__init__()
        deg = torch.bincount(edge_index[0],
                             weights=(edge_weight if edge_weight is not None
                                      else torch.ones(edge_index.shape[1])),
                             minlength=num_nodes)
        iso_score = 1.0 / (deg + 1e-3)      # alto = più isolato
        self.register_buffer("iso_score", iso_score)           # [N]
        self.gcn = GCNConv(1, hidden)
        self.dropout = dropout
        self.proj = nn.Linear(hidden, horizon)

    def forward(self, x, edge_index):                          # x: [B,N,T,1] [B,T,N,F]
        B, T, N, _ = x.shape                                    # [B,T,N,F]
        x_mean = x.mean(dim=1).squeeze(-1)                      # [B,N]
        x_weighted = x_mean * self.iso_score  # broadcast isolati
        outs = []
        for b in range(B):
            h = F.dropout(self.gcn(x_weighted[b].unsqueeze(-1), edge_index), self.dropout)  # [N,hidden]
            outs.append(h)
        outs = torch.stack(outs, dim=0)                        # [B,N,hidden]
        return self.proj(outs)                                 # [B,N,H]


class AttentiveGatingv2(nn.Module):
    def __init__(self, in_channels: int, num_experts: int,
                 k: int = 6, d_model: int = 32, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.proj  = nn.Linear(in_channels, d_model)
        self.attn  = nn.MultiheadAttention(d_model, num_heads,
                                           batch_first=True)
        self.fc    = nn.Linear(d_model, num_experts)
        self.dropout = dropout

    def forward(self, x):                      # x: [B, N, T, C]
        B, T, N, C = x.shape

        # 1) ultimi k step → [B, k, N, C]
        x_k = x[:, -self.k:, :, :]

        # 2) flatten nodo nel batch → [B*N, k, C]
        x_k = rearrange(x_k, 'b k n c -> (b n) k c')

        # 3) proiezione + self‑attention
        qkv = F.dropout(self.proj(x_k), self.dropout)                   # [B*N, k, d_model]
        h, _ = self.attn(qkv, qkv, qkv)       # attenzione temporale

        # 4) riassunto e α
        summary = F.dropout(h.mean(dim=1), self.dropout)               # [B*N, d_model]
        summary = summary.view(B, N, -1)      # [B, N, d_model]
        α = torch.softmax(self.fc(summary), dim=-1)   # [B, N, E]
        return α




# class DGM_module(nn.Module):
#     def __init__(self,hparams):
#         super().__init__()
#         conv_layers = hparams.conv_layers
#         dgm_layers = hparams.dgm_layers
#         fc_layers = hparams.fc_layers
#         self.num_nodes = hparams.num_nodes
#         self.hparams = hparams
#
#         self.graph_f = ModuleList()
#         self.node_g = ModuleList()
#         for i, (dgm_l, conv_l) in enumerate(zip(dgm_layers, conv_layers)):
#             if len(dgm_l) > 0:
#                 if hparams.ffun == 'gcn':
#                     self.graph_f.append(DGM_d(GCNConv(dgm_l[0], dgm_l[-1]),
#                                               k=hparams.k,
#                                               distance=hparams.distance,
#                                               num_nodes=hparams.num_nodes))
#                 if hparams.ffun == 'gat':
#                     self.graph_f.append(DGM_d(GATConv(dgm_l[0], dgm_l[-1]),
#                                               k=hparams.k,
#                                               distance=hparams.distance,
#                                               num_nodes=hparams.num_nodes))
#                 if hparams.ffun == 'mlp':
#                     self.graph_f.append(DGM_d(MLP(dgm_l),
#                                               k=hparams.k,
#                                               distance=hparams.distance,
#                                               num_nodes=hparams.num_nodes))
#                 if hparams.ffun == 'knn':
#                     self.graph_f.append(DGM_d(Identity(retparam=0),
#                                               k=hparams.k,
#                                               distance=hparams.distance,
#                                               num_nodes=hparams.num_nodes))
#                 if hparams.ffun == 'gcn1d':
#                     self.graph_f.append(DGM_d(GCN1DConv(dgm_l[0], dgm_l[-1], 1, 1),
#                                               k=hparams.k,
#                                               distance=hparams.distance,
#                                               num_nodes=hparams.num_nodes))
#                 if hparams.ffun == 'gcn1d-big':
#                     self.graph_f.append(DGM_d(GCN1DConv_big(dgm_l[0], dgm_l[-1], 1, 1),
#                                               k=hparams.k,
#                                               distance=hparams.distance,
#                                               num_nodes=hparams.num_nodes))
#                 if hparams.ffun == 'sts3m':
#                     self.graph_f.append(DGM_d(STS3M(dgm_l[0], dgm_l[-1]),
#                                               k=hparams.k,
#                                               distance=hparams.distance,
#                                               num_nodes=hparams.num_nodes))
#             else:
#                 self.graph_f.append(Identity())
#
#             if hparams.gfun == 'edgeconv':
#                 conv_l = conv_l.copy()
#                 conv_l[0] = conv_l[0] * 2
#                 self.node_g.append(EdgeConv(MLP(conv_l), hparams.pooling))
#             if hparams.gfun == 'gcn':
#                 self.node_g.append(GCNConv(conv_l[0], conv_l[1]))
#             if hparams.gfun == 'gat':
#                 self.node_g.append(GATConv(conv_l[0], conv_l[1]))
#             if hparams.gfun == 'gcn1d':
#                 self.node_g.append(GCN1DConv(conv_l[0], conv_l[1], 1, 1))
#             if hparams.gfun == 'gcn1d-big':
#                 self.node_g.append(GCN1DConv_big(conv_l[0], conv_l[1], 1, 1))
#             if hparams.gfun == 'sts3m':
#                 self.node_g.append(STS3M(dgm_l[0], dgm_l[-1]),
#                                               k=hparams.k,
#                                               distance=hparams.distance,
#                                               num_nodes=hparams.num_nodes)
#
#         if hparams.pre_fc is not None and len(hparams.pre_fc) > 0:
#             self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
#         self.fc = torch.nn.Linear(fc_layers[0], fc_layers[-1])
#
#     def forward(self,  x, edges=None, edges_weight=None):
#         if self.hparams.pre_fc is not None and len(self.hparams.pre_fc) > 0:  # [B, N, F]
#             x = self.pre_fc(x)
#         graph_x = x.detach()
#         lprobslist = []
#         for f, g in zip(self.graph_f, self.node_g):
#             graph_x, edges, lprobs, edges_weight = f(graph_x, edges, None, None)
#             graph_x = graph_x.squeeze()
#             x_ = torch.reshape(x, (int(x.shape[0] / self.hparams.num_nodes), self.hparams.num_nodes, -1))
#             b, n, d = x_.shape
#             self.edges = edges
#             x_pre = g(torch.dropout(x.view(-1, d), self.hparams.dropout, train=self.training), edges)
#             x = torch.nn.functional.relu(x_pre)  # .view(b, n, -1)
#             graph_x = torch.cat([graph_x, x.detach()], -1)
#             if lprobs is not None:
#                 lprobslist.append(lprobs)
#         x = F.dropout(x, p=self.hparams.dropout, training=self.training)
#         out = self.fc(x)
#         lprobs_out = torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None
#         return out, lprobs_out


# ----------  Mixture‑of‑Experts completo  ----------
class MixtureOfExperts(nn.Module):
    """
    Combina gli output dei vari expert con pesi α prodotti dal gating.
    Output: ŷ  [B,N,H] + opzionalmente i pesi α  [B,N,E] per analisi.
    """
    def __init__(self,
                 num_nodes: int,
                 lags: int,
                 horizon: int,
                 edge_index,
                 edge_weight=None,
                 hidden: int = 32,
                 num_harmonics: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        # -----  costruiamo gli expert  -----
        self.experts = nn.ModuleList([
            SeasonalExpert(lags, horizon, num_harmonics=num_harmonics, hidden=hidden),
            SpikeExpert(in_channels=1, hidden=hidden, horizon=horizon, dropout=dropout),
            IsolatedNodeExpert(num_nodes=num_nodes,
                               hidden=hidden,
                               horizon=horizon,
                               edge_index=edge_index,
                               edge_weight=edge_weight),
        ])
        self.num_experts = len(self.experts)
        # -----  gating  -----
        self.gating = AttentiveGating(in_channels=1, num_experts=self.num_experts)
        # edge_index (fisso) per l’expert IsolatedNode
        self.register_buffer("edge_index", edge_index)

    def forward(self, x):                                       # x: [B,N,T,1]
        preds = []
        for ex in self.experts:
            # IsolatedNodeExpert richiede anche edge_index
            if isinstance(ex, IsolatedNodeExpert):
                preds.append(ex(x, self.edge_index))            # [B,N,H]
            else:
                preds.append(ex(x))                             # [B,N,H]
        preds = torch.stack(preds, dim=2)                       # [B,N,E,H] (stack along expert dimension)
        α = self.gating(x)                                      # [B,N,E]
        y_hat = (α.unsqueeze(-1) * preds).sum(dim=2)            # [B,N,H]
        return y_hat.reshape((-1, y_hat.shape[-1]))


class MixtureOfExpertsDGM(nn.Module):
    """
    Combina gli output dei vari expert con pesi α prodotti dal gating.
    Output: ŷ  [B,N,H] + opzionalmente i pesi α  [B,N,E] per analisi.
    """
    def __init__(self,
                 hparams,
                 num_nodes: int,
                 lags: int,
                 horizon: int,
                 edge_index,
                 edge_weight=None,
                 hidden: int = 32,
                 num_harmonics: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        # -----  costruiamo gli expert  -----
        self.experts = nn.ModuleList([
            SeasonalExpertv2(lags, horizon, num_harmonics=num_harmonics, hidden=hidden, dropout=hparams.dropout),
            SpikeExpertv2(in_channels=1, hidden=hidden, horizon=horizon, dropout=hparams.dropout),
            IsolatedNodeExpertv2(num_nodes=num_nodes,
                               hidden=hidden,
                               horizon=horizon,
                               edge_index=edge_index,
                               edge_weight=edge_weight,
                               dropout=hparams.dropout),
            DGM_module(hparams)
        ])
        self.num_experts = len(self.experts)
        # -----  gating  -----
        self.gating = AttentiveGatingv2(in_channels=1, num_experts=self.num_experts, dropout=hparams.dropout)
        # edge_index (fisso) per l’expert IsolatedNode
        self.register_buffer("edge_index", edge_index)

    def forward(self, x, edge_index=None, edge_weight=None):                                       # x: [B,N,T,1]
        preds = []
        lproblist = []
        for ex in self.experts:
            # IsolatedNodeExpert richiede anche edge_index
            if isinstance(ex, IsolatedNodeExpert):
                preds.append(ex(x, self.edge_index))            # [B,N,H]
            elif isinstance(ex, DGM_module):
                out, lprobs = ex(rearrange(x, 'b t n f -> (b n) (t f)'), self.edge_index)
                preds.append(out.reshape(x.shape[0], x.shape[2], -1))
                lproblist.append(lprobs)
            else:
                preds.append(ex(x))                             # [B,N,H]
        preds = torch.stack(preds, dim=2)                       # [B,N,E,H]
        α = self.gating(x)                                      # [B,N,E]
        y_hat = (α.unsqueeze(-1) * preds).sum(dim=2)            # [B,N,H]
        return y_hat.reshape((-1, y_hat.shape[-1])), lproblist


if __name__ == "__main__":
    # Grafo toy: linea di 5 nodi (0‑1‑2‑3‑4)
    edge_index = torch.tensor([[0, 1, 2, 3],
                               [1, 2, 3, 4]], dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # make undirected

    B, N, T, H = 4, 5, 48, 12
    x = torch.randn(B, N, T, 1)  # batch fittizio

    model = MixtureOfExperts(num_nodes=N,
                             lags=T,
                             horizon=H,
                             edge_index=edge_index)

    y_hat, gate = model(x)
    print("Predictions:", y_hat.shape)   # [4,5,12]
    print("Gating α:", gate.shape)       # [4,5,3]

    # ----------------  mini‑training loop  ----------------
    target = torch.randn(B, N, H)        # ground truth fittizia
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for step in range(10):
        optim.zero_grad()
        pred, _ = model(x)
        loss = loss_fn(pred, target)
        loss.backward()
        optim.step()
        if (step + 1) % 2 == 0:
            print(f"step {step+1} | loss = {loss.item():.4f}")
