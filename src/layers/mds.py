from einops import rearrange, repeat, einsum
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
from torch_geometric.utils import to_dense_adj
from knn_cuda import KNN

from src.config import Parameters
from src.layers.gcn1d import GCN1DConv, GCN1DConv_big
from src.layers.utils import pairwise_poincare_distances, pairwise_euclidean_distances


class MLP(nn.Module):
    def __init__(self, layers_size, final_activation=False, dropout=0):
        super(MLP, self).__init__()
        layers = []
        for li in range(1, len(layers_size)):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layers_size[li - 1], layers_size[li]))
            if li == len(layers_size) - 1 and not final_activation:
                continue
            layers.append(nn.LeakyReLU(0.1))

        self.MLP = nn.Sequential(*layers)

    def forward(self, x, e=None):
        x = self.MLP(x)
        return x


class Identity(nn.Module):
    def __init__(self, retparam=None):
        self.retparam = retparam
        super(Identity, self).__init__()

    def forward(self, *params):
        if self.retparam is not None:
            return params[self.retparam]
        return params


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class STS3M(nn.Module):
    def __init__(self, args: Parameters, dim_in:int, dim_out:int):
        super().__init__()
        self.args = args
        self.args.__post_init__()
        self.in_proj = nn.Linear(dim_in, args.d_inner * 2, bias=args.bias)  # dim_in al posto di d_model
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, dim_out, bias=args.bias)
        self.num_nodes = 0

    def forward(self, x, adj_matrix):
        if adj_matrix.shape[0] == 2:
            if adj_matrix.dtype != torch.int64:
                adj_matrix = adj_matrix.to(torch.int64)
            adj_matrix = to_dense_adj(adj_matrix, max_num_nodes=self.args.num_nodes).squeeze(0)
        adj_matrix = adj_matrix.to(torch.int64)
        adj_matrix = adj_matrix.detach()
        x = rearrange(x, 'b n t f -> (b n) t f')
        (b, l, d) = x.shape  # (BN, T, F)
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        y = self.ssm(x, adj_matrix)
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    def ssm(self, x, adj_matrix):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        y = self.selective_scan(x, delta, A, B, C, D, adj_matrix)
        return y

    def selective_scan(self, u, delta, A, B, C, D, adj_matrix):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        # This is the new version of Selective Scan Algorithm named as "Graph Selective Scan"
        # In Graph Selective Scan, we use the Feed-Forward graph information from KFGN, and incorporate the Feed-Forward information with "delta"
        # temp_adj = self.kfgn.gc_list[-1].get_transformed_adjacency()
        adj_matrix = to_dense_adj(adj_matrix, max_num_nodes=self.args.num_nodes).squeeze(0)
        temp_adj_padded = torch.ones(d_in, d_in, device=adj_matrix.device)
        temp_adj_padded[:adj_matrix.size(0), :adj_matrix.size(1)] = adj_matrix
        delta_p = torch.matmul(delta, temp_adj_padded)

        # The fused param delta_p will participate in the following upgrading of deltaA and deltaB_u
        deltaA = torch.exp(einsum(delta_p, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta_p, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = (deltaA[:, i] * x + deltaB_u[:, i]).detach()  # TODO: check peggiormento performance
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in').detach()  # TODO: check peggiormento performance
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        y = y + u * D

        return y


class DGM_d(nn.Module):
    def __init__(self, embed_f, k=5, distance="euclidean", num_nodes=1, sparse=True, hparams=None):
        super(DGM_d, self).__init__()

        self.sparse = sparse
        self.temperature = nn.Parameter(torch.tensor(1. if distance == "hyperbolic" else 4.).float())
        self.embed_f = embed_f
        self.centroid = None
        self.scale = None
        self.k = k
        self.distance = distance
        self.num_of_nodes = 0
        self.num_nodes = num_nodes
        self.hparams = hparams

        self.debug = False

    def forward(self, x_pre, A, not_used=None, fixedges=None):
        # Preprocess
        b, t, n, f = x_pre.shape

        if self.hparams.ffun == 'sts3m':
            x_pre = rearrange(x_pre, 'b t n f -> b n t f')
            x = self.embed_f(x_pre, A)  # out.shape = (bn t f)
            x = torch.reshape(x, (b, n, t, -1))
        elif self.hparams.ffun == 'gcn':
            # x_pre = rearrange(x_pre, 'b t n f -> b n t f')
            # x_pre = torch.mean(x_pre, dim=2)
            # x_pre = rearrange(x_pre, 'b n f -> (b n) f')
            x = self.embed_f(x_pre, A)  # out.shape = (bn t f)
            # x = torch.reshape(x, (b, n, -1))
            # x = torch.repeat_interleave(x, repeats=t, dim=2)
            # x = torch.reshape(x, (b, n, t, -1))

        if self.training:
            if fixedges is not None:
                return x, fixedges, torch.zeros(fixedges.shape[0], fixedges.shape[-1] // self.k, self.k,
                                                dtype=torch.float, device=x.device)
            # sampling here
            edges_hat, logprobs = self.sample_without_replacement(x)

        else:
            with torch.no_grad():
                if fixedges is not None:
                    return x, fixedges, torch.zeros(fixedges.shape[0], fixedges.shape[-1] // self.k, self.k,
                                                    dtype=torch.float, device=x.device)
                # sampling here
                edges_hat, logprobs = self.sample_without_replacement(x)

        if self.debug:
            if self.distance == "euclidean":
                D, _x = pairwise_euclidean_distances(x)
            if self.distance == "hyperbolic":
                D, _x = pairwise_poincare_distances(x)

            self.D = (D * torch.exp(torch.clamp(self.temperature, -5, 5))).detach().cpu()
            self.edges_hat = edges_hat.detach().cpu()
            self.logprobs = logprobs.detach().cpu()
            self.x = x

        return x, edges_hat, logprobs, None


    def sample_without_replacement_not_so_old(self, x):
        x = torch.reshape(x[0,:,:], (-1, self.num_nodes, x.shape[2]))
        #TODO: probabilmente dovrai fare un flatten da x=(b,n,t,f) a x=(b,n,f)
        # x = x.mean(dim=2)
        b, n, _ = x.shape

        if self.distance == "euclidean":
            G_i = x.clone().unsqueeze(2)
            X_j = x.clone().unsqueeze(1)
            mD = ((G_i - X_j) ** 2).sum(-1)

            # argKmin already add gumbel noise
            lq = mD * torch.exp(torch.clamp(self.temperature, -5, 5))

            # batch-mode modification
            lq = torch.sum(lq, dim=0)
            knn = KNN(k=self.k, transpose_mode=True)
            dist, indices = knn(lq.unsqueeze(0), lq.unsqueeze(0))
            x1 = torch.gather(x, -2,
                              indices.view(indices.unsqueeze(0).shape[0], -1)[..., None].repeat(1, 1, x.shape[-1]))
            x2 = x[:, :, None, :].repeat(1, 1, self.k, 1).view(x.shape[0], -1, x.shape[-1])
            logprobs = (-(x1 - x2).pow(2).sum(-1) * torch.exp(torch.clamp(self.temperature, -5, 5))).reshape(x.shape[0],
                                                                                                             -1, self.k)

        if self.distance == "hyperbolic":
            pass
            x_norm = (x ** 2).sum(-1, keepdim=True)
            x_norm = (x_norm.sqrt() - 1).relu() + 1
            x = x / (x_norm * (1 + 1e-2))  # safe distance to the margin
            x_norm = (x ** 2).sum(-1, keepdim=True)

            G_i = torch.tensor(x[:, :, None, :])  # (M**2, 1, 2)
            X_j = torch.tensor(x[:, None, :, :])  # (1, N, 2)

            G_i2 = torch.tensor(1 - x_norm[:, :, None, :])  # (M**2, 1, 2)
            X_j2 = torch.tensor(1 - x_norm[:, None, :, :])  # (1, N, 2)

            pq = ((G_i - X_j) ** 2).sum(-1)
            N = (G_i2 * X_j2)
            XX = (1e-6 + 1 + 2 * pq / N)
            mD = (XX + (XX ** 2 - 1).sqrt()).log() ** 2

            lq = mD * torch.exp(torch.clamp(self.temperature, -5, 5))

            indices = torch.argmin(lq, dim=1)[:, :self.k]  # lq.argKmin(self.k, dim=1)

            x1 = torch.gather(x, -2, indices.view(indices.shape[0], -1)[..., None].repeat(1, 1, x.shape[-1]))
            x2 = x[:, :, None, :].repeat(1, 1, self.k, 1).view(x.shape[0], -1, x.shape[-1])

            x1_n = torch.gather(x_norm, -2,
                                indices.view(indices.shape[0], -1)[..., None].repeat(1, 1, x_norm.shape[-1]))
            x2_n = x_norm[:, :, None, :].repeat(1, 1, self.k, 1).view(x.shape[0], -1, x_norm.shape[-1])

            pq = (x1 - x2).pow(2).sum(-1)
            pqn = ((1 - x1_n) * (1 - x2_n)).sum(-1)
            XX = 1e-6 + 1 + 2 * pq / pqn
            dist = torch.log(XX + (XX ** 2 - 1).sqrt()) ** 2
            logprobs = (-dist * torch.exp(torch.clamp(self.temperature, -5, 5))).reshape(x.shape[0], -1, self.k)

            if self.debug:
                self._x = x.detach().cpu() + 0

        # rows = torch.arange(n).view(1, n, 1).to(x.device).repeat(b, 1, self.k)
        # edges = torch.stack((indices.view(b, -1), rows.view(b, -1)), -2)

        # Modifica per batches
        single_batch_src = torch.arange(n).view(n, 1).repeat(1, self.k).view(-1).to('cuda')
        single_batch_tfg = indices.view(-1).to('cuda')
        single_batch_src = single_batch_src.repeat(b)
        single_batch_tfg = single_batch_tfg.repeat(b)
        tensor_to_adapt_idx_to_batch_size = torch.arange(b) * n
        tensor_to_adapt_idx_to_batch_size = tensor_to_adapt_idx_to_batch_size.unsqueeze(-1).repeat(1, n*self.k).view(-1).to('cuda')
        single_batch_src += tensor_to_adapt_idx_to_batch_size
        single_batch_tfg += tensor_to_adapt_idx_to_batch_size
        edges = torch.cat([single_batch_src.unsqueeze(0), single_batch_tfg.unsqueeze(0)], dim=0)
        return edges, logprobs


    def sample_without_replacement(self, x):
        x = torch.reshape(x[0,:,:], (-1, self.num_nodes, x.shape[2]))
        #TODO: probabilmente dovrai fare un flatten da x=(b,n,t,f) a x=(b,n,f)
        # x = x.mean(dim=2)
        # TODO: ripartire da qui, prova con CLaude ad adattare DGM a più grafi con edge_index diversi
        b, n, _ = x.shape

        if self.distance == "euclidean":
            G_i = x.clone().unsqueeze(2)
            X_j = x.clone().unsqueeze(1)
            mD = ((G_i - X_j) ** 2).sum(-1)

            # argKmin already add gumbel noise
            lq = mD * torch.exp(torch.clamp(self.temperature, -5, 5))

            # batch-mode modification
            lq = torch.sum(lq, dim=0)
            knn = KNN(k=self.k, transpose_mode=True)
            dist, indices = knn(lq.unsqueeze(0), lq.unsqueeze(0))
            x1 = torch.gather(x, -2,
                              indices.view(indices.unsqueeze(0).shape[0], -1)[..., None].repeat(1, 1, x.shape[-1]))
            x2 = x[:, :, None, :].repeat(1, 1, self.k, 1).view(x.shape[0], -1, x.shape[-1])
            logprobs = (-(x1 - x2).pow(2).sum(-1) * torch.exp(torch.clamp(self.temperature, -5, 5))).reshape(x.shape[0],
                                                                                                             -1, self.k)

        if self.distance == "hyperbolic":
            pass
            x_norm = (x ** 2).sum(-1, keepdim=True)
            x_norm = (x_norm.sqrt() - 1).relu() + 1
            x = x / (x_norm * (1 + 1e-2))  # safe distance to the margin
            x_norm = (x ** 2).sum(-1, keepdim=True)

            G_i = torch.tensor(x[:, :, None, :])  # (M**2, 1, 2)
            X_j = torch.tensor(x[:, None, :, :])  # (1, N, 2)

            G_i2 = torch.tensor(1 - x_norm[:, :, None, :])  # (M**2, 1, 2)
            X_j2 = torch.tensor(1 - x_norm[:, None, :, :])  # (1, N, 2)

            pq = ((G_i - X_j) ** 2).sum(-1)
            N = (G_i2 * X_j2)
            XX = (1e-6 + 1 + 2 * pq / N)
            mD = (XX + (XX ** 2 - 1).sqrt()).log() ** 2

            lq = mD * torch.exp(torch.clamp(self.temperature, -5, 5))

            indices = torch.argmin(lq, dim=1)[:, :self.k]  # lq.argKmin(self.k, dim=1)

            x1 = torch.gather(x, -2, indices.view(indices.shape[0], -1)[..., None].repeat(1, 1, x.shape[-1]))
            x2 = x[:, :, None, :].repeat(1, 1, self.k, 1).view(x.shape[0], -1, x.shape[-1])

            x1_n = torch.gather(x_norm, -2,
                                indices.view(indices.shape[0], -1)[..., None].repeat(1, 1, x_norm.shape[-1]))
            x2_n = x_norm[:, :, None, :].repeat(1, 1, self.k, 1).view(x.shape[0], -1, x_norm.shape[-1])

            pq = (x1 - x2).pow(2).sum(-1)
            pqn = ((1 - x1_n) * (1 - x2_n)).sum(-1)
            XX = 1e-6 + 1 + 2 * pq / pqn
            dist = torch.log(XX + (XX ** 2 - 1).sqrt()) ** 2
            logprobs = (-dist * torch.exp(torch.clamp(self.temperature, -5, 5))).reshape(x.shape[0], -1, self.k)

            if self.debug:
                self._x = x.detach().cpu() + 0

        # rows = torch.arange(n).view(1, n, 1).to(x.device).repeat(b, 1, self.k)
        # edges = torch.stack((indices.view(b, -1), rows.view(b, -1)), -2)

        # Modifica per batches
        single_batch_src = torch.arange(n).view(n, 1).repeat(1, self.k).view(-1).to('cuda')
        single_batch_tfg = indices.view(-1).to('cuda')
        single_batch_src = single_batch_src.repeat(b)
        single_batch_tfg = single_batch_tfg.repeat(b)
        tensor_to_adapt_idx_to_batch_size = torch.arange(b) * n
        tensor_to_adapt_idx_to_batch_size = tensor_to_adapt_idx_to_batch_size.unsqueeze(-1).repeat(1, n*self.k).view(-1).to('cuda')
        single_batch_src += tensor_to_adapt_idx_to_batch_size
        single_batch_tfg += tensor_to_adapt_idx_to_batch_size
        edges = torch.cat([single_batch_src.unsqueeze(0), single_batch_tfg.unsqueeze(0)], dim=0)
        return edges, logprobs

class DGMmodule(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        conv_layers = hparams.conv_layers
        dgm_layers = hparams.dgm_layers
        fc_layers = hparams.fc_layers
        hparams.num_nodes = hparams.ndim
        self.num_nodes = hparams.num_nodes
        self.hparams = hparams

        self.graph_f = ModuleList()
        self.node_g = ModuleList()

        # self.graph_f, self.node_g = self.compose_g_and_f()
        # def compose_g_and_f(self, dgm_layers, conv_layers, hparams):

        for i, (dgm_l, conv_l) in enumerate(zip(dgm_layers, conv_layers)):
            if len(dgm_l) > 0:
                if hparams.ffun == 'gcn':
                    self.graph_f.append(DGM_d(GCNConv(dgm_l[0], dgm_l[-1]),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes,
                                              hparams=hparams))
                if hparams.ffun == 'gat':
                    self.graph_f.append(DGM_d(GATConv(dgm_l[0], dgm_l[-1]),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes,
                                        hparams=hparams))
                if hparams.ffun == 'mlp':
                    self.graph_f.append(DGM_d(MLP(dgm_l),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes,
                                        hparams=hparams))
                if hparams.ffun == 'knn':
                    self.graph_f.append(DGM_d(Identity(retparam=0),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes,
                                        hparams=hparams))
                if hparams.ffun == 'gcn1d':
                    self.graph_f.append(DGM_d(GCN1DConv(dgm_l[0], dgm_l[-1], 1, 1),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes,
                                        hparams=hparams))
                if hparams.ffun == 'gcn1d-big':
                    self.graph_f.append(DGM_d(GCN1DConv_big(dgm_l[0], dgm_l[-1], 1, 1),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes,
                                        hparams=hparams))
                if hparams.ffun == 'sts3m':
                    self.graph_f.append(DGM_d(STS3M(hparams,dgm_l[0], dgm_l[-1]),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes,
                                        hparams=hparams))
            else:
                self.graph_f.append(Identity())

            if hparams.gfun == 'edgeconv':
                conv_l = conv_l.copy()
                conv_l[0] = conv_l[0] * 2
                self.node_g.append(EdgeConv(MLP(conv_l), hparams.pooling))
            if hparams.gfun == 'gcn':
                self.node_g.append(GCNConv(conv_l[0], conv_l[1]))
            if hparams.gfun == 'gat':
                self.node_g.append(GATConv(conv_l[0], conv_l[1]))
            if hparams.gfun == 'gcn1d':
                self.node_g.append(GCN1DConv(conv_l[0], conv_l[1], 1, 1))
            if hparams.gfun == 'gcn1d-big':
                self.node_g.append(GCN1DConv_big(conv_l[0], conv_l[1], 1, 1))
            if hparams.gfun == 'sts3m':
                self.node_g.append(STS3M(hparams,conv_l[0], conv_l[1]))

        # Prediction
        if hparams.pre_fc is not None and len(hparams.pre_fc) > 0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        self.fc = torch.nn.Linear(fc_layers[0], fc_layers[-1])

    def forward(self,  x, edges=None, edges_weight=None):
        # Preprocessing
        if self.hparams.pre_fc is not None and len(self.hparams.pre_fc) > 0:  # [B, N, F]
            x = self.pre_fc(x)

        # Dual-step f and g
        graph_x = x.detach()
        lprobslist = []
        for f, g in zip(self.graph_f, self.node_g):
            # Graph learning
            graph_x, edges, lprobs, edges_weight = f(graph_x, edges, None, None)
            edges = edges.detach()
            if not isinstance(f, Identity):
                graph_x = rearrange(graph_x, 'b n t d -> b t n d')
            # self.edges = edges

            # Diffusion
            x = F.relu(g(torch.dropout(x, self.hparams.dropout, train=self.training), edges))
            x = torch.reshape(x, (graph_x.shape[0], graph_x.shape[1], x.shape[1], x.shape[2]))
            graph_x = torch.cat([graph_x, x.detach()], -1)  # join x_f e x_g
            if lprobs is not None:
                lprobslist.append(lprobs)

        # Prediction
        x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        out = self.fc(x)
        lprobs_out = torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None
        return out, lprobs_out


class AttentiveGating(nn.Module):
    def __init__(self, hparams: Parameters, num_experts: int):
        super().__init__()
        self.hparams = hparams
        self.proj  = nn.Linear(self.hparams.channels,
                               self.hparams.d_model_moe)
        self.attn  = nn.MultiheadAttention(self.hparams.d_model_moe,
                                           self.hparams.num_heads_moe,
                                           batch_first=True)
        self.fc    = nn.Linear(self.hparams.d_model_moe,
                               num_experts)
        self.dropout = self.hparams.dropout_moe
        self.k = self.hparams.k_moe

    def forward(self, x):                                      # x: [B, N, T, C]
        B, T, N, C = x.shape
        x_k = x[:, -self.k:, :, :]                             # 1) ultimi k step → [B, k, N, C]
        x_k = rearrange(x_k, 'b k n c -> (b n) k c')   # 2) flatten nodo nel batch → [B*N, k, C]
        qkv = F.dropout(self.proj(x_k), self.dropout)          # 3) proiezione + self‑attention, [B*N, k, d_model]
        h, _ = self.attn(qkv, qkv, qkv)                        # attenzione temporale
        summary = F.dropout(h.mean(dim=1), self.dropout)      # 4) riassunto e α, [B*N, d_model]
        summary = summary.view(B, N, -1)      # [B, N, d_model]
        α = torch.softmax(self.fc(summary), dim=-1)   # [B, N, E]
        return α

class MixtureOfExpertsDgmSsm(nn.Module):
    """
    Combina gli output dei vari expert con pesi α prodotti dal gating.
    Output: ŷ  [B,N,H] + opzionalmente i pesi α  [B,N,E] per analisi.
    """
    def __init__(self, hparams):
        super().__init__()
        # List of experts
        self.experts = nn.ModuleList([
            DGMmodule(hparams),
            DGMmodule(hparams),
            DGMmodule(hparams)])

        # Gating mechanism
        self.gating = AttentiveGating(hparams=hparams, num_experts=len(self.experts))

    def forward(self, x, edge_index=None, edge_weight=None):
        """
        Args:
            x (Tensor): input of shape [B, N, T,1]
            edge_index (Tensor): input of shape [2, E]
            edge_weight (Tensor or None): input of shape [E]
        """

        # Expert loop
        preds = []
        lproblist = []
        for ex in self.experts:
            out, lprobs = ex(x, edge_index)  # prima invece di x rearrange(x, 'b t n f -> (b n) (t f)')
            preds.append(out.reshape(x.shape[0], x.shape[2], -1))
            lproblist.append(lprobs)

        # Gating and prediction
        preds = torch.stack(preds, dim=2)                       # [B,N,E,H]
        α = self.gating(x)                                      # [B,N,E]
        y_hat = (α.unsqueeze(-1) * preds).sum(dim=2)            # [B,N,H]
        return y_hat.reshape((-1, y_hat.shape[-1])), lproblist
# class MixtureOfExpertsDgmSsm(nn.Module):
#     """
#     3‑experts MoE con forward in parallelo via CUDA Streams.
#     Restituisce:
#         y_hat  [B, N, H]  o [B*N, H] se flat_output=True
#         lprobs list[len=E] – diagnostica per ogni expert
#         alpha  [B, N, E]   – opzionale (return_alpha=True)
#     """
#     def __init__(self, hparams, flat_output: bool = False):
#         super().__init__()
#         self.flat_output = flat_output
#
#         # === 1. Creiamo i 3 esperti ========================================
#         self.experts = nn.ModuleList([DGMmodule(hparams) for _ in range(3)])
#
#         # === 2. Gating ======================================================
#         self.gating = AttentiveGating(hparams=hparams, num_experts=len(self.experts))
#
#     def _run_expert(self, ex, x, edge_index, edge_weight):
#         """Helper per uniformare le chiamate."""
#         out, lprobs = ex(x, edge_index, edge_weight)   # out [B,N,H]
#         return out, lprobs
#
#     def forward(
#         self,
#         x: torch.Tensor,               # [B, N, T, 1]
#         edge_index: torch.Tensor | None = None,
#         edge_weight: torch.Tensor | None = None,
#         return_alpha: bool = False,
#     ):
#         B, _, T, _ = x.shape
#         device   = x.device
#         E        = len(self.experts)
#
#         # ===========================================================
#         # 1) PARALLELIZZAZIONE DEGLI EXPERT
#         # ===========================================================
#         preds   = [None] * E
#         lprobs  = [None] * E
#
#         # Caso GPU – uno stream dedicato per expert
#         if device.type == "cuda":
#             streams = [torch.cuda.Stream(device=device) for _ in range(E)]
#
#             # Lancia tutti gli expert in stream separati (overlap di kernel)
#             for i, (ex, st) in enumerate(zip(self.experts, streams)):
#                 with torch.cuda.stream(st):
#                     preds[i], lprobs[i] = self._run_expert(ex, x, edge_index, edge_weight)
#
#             # Sincronizza tutti gli stream prima di proseguire
#             torch.cuda.synchronize(device)
#
#         # Caso CPU / fallback seriale
#         else:
#             for i, ex in enumerate(self.experts):
#                 preds[i], lprobs[i] = self._run_expert(ex, x, edge_index, edge_weight)
#
#         #             preds.append(out.reshape(x.shape[0], x.shape[2], -1))
#         #             lproblist.append(lprobs)
#         #
#         #         # Gating and prediction
#         #         preds = torch.stack(preds, dim=2)                       # [B,N,E,H]
#         #         α = self.gating(x)                                      # [B,N,E]
#         #         y_hat = (α.unsqueeze(-1) * preds).sum(dim=2)            # [B,N,H]
#         #         return y_hat.reshape((-1, y_hat.shape[-1])), lproblist
#         preds = [torch.reshape(p, (B,T,-1)) for p in preds]
#         preds = torch.stack(preds, dim=2)
#
#
#         # ===========================================================
#         # 2) GATING E COMBINAZIONE
#         # ===========================================================
#         alpha  = self.gating(x)                       # [B,N,E]
#         y_hat = (alpha.unsqueeze(-1) * preds).sum(dim=2)            # [B,N,H]
#         return y_hat.reshape((-1, y_hat.shape[-1])), lprobs
