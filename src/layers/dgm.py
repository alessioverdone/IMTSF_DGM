import torch
from torch import nn
import torch.nn.functional as F

from knn_cuda import KNN
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv, GATConv, EdgeConv

from src.layers.gcn1d import GCN1DConv, GCN1DConv_big
from src.layers.mds import STS3M
from src.layers.utils import pairwise_euclidean_distances, pairwise_poincare_distances


class DGM_module(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        conv_layers = hparams.conv_layers
        dgm_layers = hparams.dgm_layers
        fc_layers = hparams.fc_layers
        self.num_nodes = hparams.num_nodes
        self.hparams = hparams

        self.graph_f = ModuleList()
        self.node_g = ModuleList()
        for i, (dgm_l, conv_l) in enumerate(zip(dgm_layers, conv_layers)):
            if len(dgm_l) > 0:
                if hparams.ffun == 'gcn':
                    self.graph_f.append(DGM_d(GCNConv(dgm_l[0], dgm_l[-1]),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
                if hparams.ffun == 'gat':
                    self.graph_f.append(DGM_d(GATConv(dgm_l[0], dgm_l[-1]),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
                if hparams.ffun == 'mlp':
                    self.graph_f.append(DGM_d(MLP(dgm_l),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
                if hparams.ffun == 'knn':
                    self.graph_f.append(DGM_d(Identity(retparam=0),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
                if hparams.ffun == 'gcn1d':
                    self.graph_f.append(DGM_d(GCN1DConv(dgm_l[0], dgm_l[-1], 1, 1),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
                if hparams.ffun == 'gcn1d-big':
                    self.graph_f.append(DGM_d(GCN1DConv_big(dgm_l[0], dgm_l[-1], 1, 1),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
                if hparams.ffun == 'sts3m':
                    self.graph_f.append(DGM_d(STS3M(dgm_l[0], dgm_l[-1]),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
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
                self.node_g.append(STS3M(dgm_l[0], dgm_l[-1]),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes)

        if hparams.pre_fc is not None and len(hparams.pre_fc) > 0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        self.fc = torch.nn.Linear(fc_layers[0], fc_layers[-1])

    def forward(self,  x, edges=None, edges_weight=None):
        if self.hparams.pre_fc is not None and len(self.hparams.pre_fc) > 0:  # [B, N, F]
            x = self.pre_fc(x)
        graph_x = x.detach()
        lprobslist = []
        for f, g in zip(self.graph_f, self.node_g):
            graph_x, edges, lprobs, edges_weight = f(graph_x, edges, None, None)
            graph_x = graph_x.squeeze()
            x_ = torch.reshape(x, (int(x.shape[0] / self.hparams.num_nodes), self.hparams.num_nodes, -1))
            b, n, d = x_.shape
            self.edges = edges
            x_pre = g(torch.dropout(x.view(-1, d), self.hparams.dropout, train=self.training), edges)
            x = torch.nn.functional.relu(x_pre)  # .view(b, n, -1)
            graph_x = torch.cat([graph_x, x.detach()], -1)
            if lprobs is not None:
                lprobslist.append(lprobs)
        x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        out = self.fc(x)
        lprobs_out = torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None
        return out, lprobs_out


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

    def forward(self, x, A, not_used=None, fixedges=None):
        # Get dimension and preprocess X
        if x.shape[0] == 1:
            x = x[0]
        if self.hparams is not None and self.hparams.ffun == 'gconvLSTM':
            batch = int(x.shape[0]/self.hparams.num_nodes)
            x = x.unsqueeze(-1)
            x = torch.reshape(x, (batch, self.hparams.num_nodes,  x.shape[1]))
        # NN
        # print(x.shape)
        x = self.embed_f(x, A)
        # print(x.shape)

        if self.hparams is not None and self.hparams.ffun == 'gconvLSTM':
            x = x.squeeze()
        if x.dim() == 2:
            x = x[None, ...]

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

    def sample_without_replacement_old(self, x):

        b, n, _ = x.shape

        if self.distance == "euclidean":
            G_i = x.clone().unsqueeze(2)
            X_j =x.clone().unsqueeze(1)


            mD = ((G_i - X_j) ** 2).sum(-1)

            # argKmin already add gumbel noise
            lq = mD * torch.exp(torch.clamp(self.temperature, -5, 5))
            knn = KNN(k=self.k, transpose_mode=True)
            dist, indices = knn(lq, lq)

            x1 = torch.gather(x, -2,
                              indices.view(indices.unsqueeze(0).shape[0], -1)[..., None].repeat(1, 1, x.shape[-1]))
            x2 = x[:, :, None, :].repeat(1, 1, self.k, 1).view(x.shape[0], -1, x.shape[-1])
            # logprobs = (-(x1 - x2).pow(2).sum(-1) * torch.exp(torch.clamp(self.temperature, -5, 5))).reshape(x.shape[0],
            #                                                                                                  -1, self.k)
            logprobs = (-(x1 - x2).pow(2).sum(-1) * torch.clamp(self.temperature, -5, 5)).reshape(x.shape[0],
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

        rows = torch.arange(n).view(1, n, 1).to(x.device).repeat(b, 1, self.k)
        edges = torch.stack((indices.view(b, -1), rows.view(b, -1)), -2)

        if self.sparse:
            return (edges + (torch.arange(b).to(x.device) * n)[:, None, None]).transpose(0, 1).reshape(2, -1), logprobs
        return edges, logprobs

    def sample_without_replacement(self, x):

        x = torch.reshape(x[0,:,:], (-1, self.num_nodes, x.shape[2]))
        b, n, _ = x.shape

        if self.distance == "euclidean":
            G_i = x.clone().unsqueeze(2)
            X_j = x.clone().unsqueeze(1)
            # G_i = torch.tensor(x[:, :, None, :])  # (M**2, 1, 2)
            # X_j = torch.tensor(x[:, None, :, :])  # (1, N, 2)

            mD = ((G_i - X_j) ** 2).sum(-1)

            # argKmin already add gumbel noise
            lq = mD * torch.exp(torch.clamp(self.temperature, -5, 5))
            # batch-mode modification
            lq = torch.sum(lq, dim=0)
            knn = KNN(k=self.k, transpose_mode=True)
            dist, indices = knn(lq.unsqueeze(0), lq.unsqueeze(0))
            # lq = lq.cpu().type(torch.float32)
            # D=2708
            # K=5
            # formula = "SqDist(x,y)"  # Use a simple Euclidean (squared) norm
            # variables = [
            #     "x = Vi(" + str(D) + ")",  # First arg : i-variable, of size D
            #     "y = Vj(" + str(D) + ")",
            # ]  # Second arg: j-variable, of size D
            # # N.B.: The number K is specified as an optional argument `opt_arg`
            # my_routine = Genred(formula, variables, reduction_op="ArgKMin", axis=1, opt_arg=K)
            # indices = my_routine(lq, lq, backend="CPU").to('cuda')
            # nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(lq.squeeze().cpu().numpy())
            # distances, indices = nbrs.kneighbors(lq.squeeze().cpu().numpy())
            # indices = torch.tensor(indices, device='cuda')
            # indices = KNN(lq, lq, 5)
            # indices = torch.argmin(lq, dim=1)[:, :self.k]  # lq.argKmin(self.k, dim=1)

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


        # if self.sparse:
        #     return (edges + (torch.arange(b).to(x.device) * n)[:, None, None]).transpose(0, 1).reshape(2, -1), logprobs
        return edges, logprobs


class DGM_d_dense(nn.Module):
    def __init__(self, embed_f, k=5, distance=pairwise_euclidean_distances, sparse=True):
        super(DGM_d_dense, self).__init__()

        self.sparse = sparse

        self.temperature = nn.Parameter(torch.tensor(1. if distance == "hyperbolic" else 4.).float())
        self.embed_f = embed_f
        self.centroid = None
        self.scale = None
        self.k = k

        self.debug = False
        if distance == 'euclidean':
            self.distance = pairwise_euclidean_distances
        else:
            self.distance = pairwise_poincare_distances

    def forward(self, x, A, not_used=None, fixedges=None):
        x = self.embed_f(x, A)

        if self.training:
            if fixedges is not None:
                return x, fixedges, torch.zeros(fixedges.shape[0], fixedges.shape[-1] // self.k, self.k,
                                                dtype=torch.float, device=x.device)

            D, _x = self.distance(x)

            # sampling here
            edges_hat, logprobs = self.sample_without_replacement(D)

        else:
            with torch.no_grad():
                if fixedges is not None:
                    return x, fixedges, torch.zeros(fixedges.shape[0], fixedges.shape[-1] // self.k, self.k,
                                                    dtype=torch.float, device=x.device)
                D, _x = self.distance(x)

                # sampling here
                edges_hat, logprobs = self.sample_without_replacement(D)

        if self.debug:
            self.D = D
            self.edges_hat = edges_hat
            self.logprobs = logprobs
            self.x = x

        return x, edges_hat, logprobs, None

    def sample_without_replacement(self, logits):
        b, n, _ = logits.shape
        #         logits = logits * torch.exp(self.temperature*10)
        logits = logits * torch.exp(torch.clamp(self.temperature, -5, 5))

        q = torch.rand_like(logits) + 1e-8
        lq = (logits - torch.log(-torch.log(q)))
        logprobs, indices = torch.topk(-lq, self.k)

        rows = torch.arange(n).view(1, n, 1).to(logits.device).repeat(b, 1, self.k)
        edges = torch.stack((indices.view(b, -1), rows.view(b, -1)), -2)

        if self.sparse:
            return (edges + (torch.arange(b).to(logits.device) * n)[:, None, None]).transpose(0, 1).reshape(2,
                                                                                                            -1), logprobs
        return edges, logprobs


class DGM_c(nn.Module):
    input_dim = 4
    debug = False

    def __init__(self, embed_f, k=None, distance="euclidean"):
        super(DGM_c, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(1).float())
        self.threshold = nn.Parameter(torch.tensor(0.5).float())
        self.embed_f = embed_f
        self.centroid = None
        self.scale = None
        self.distance = distance

        self.scale = nn.Parameter(torch.tensor(-1).float(), requires_grad=False)
        self.centroid = nn.Parameter(torch.zeros((1, 1, DGM_c.input_dim)).float(), requires_grad=False)

    def forward(self, x, A, not_used=None, fixedges=None, edges_weight=None):

        x = self.embed_f(x, A)

        # estimate normalization parameters
        if self.scale < 0:
            self.centroid.data = x.mean(-2, keepdim=True).detach()
            self.scale.data = (0.9 / (x - self.centroid).abs().max()).detach()

        if self.distance == "hyperbolic":
            D, _x = pairwise_poincare_distances((x - self.centroid) * self.scale)
        else:
            D, _x = pairwise_euclidean_distances((x - self.centroid) * self.scale)

        adj = torch.sigmoid(self.temperature * (self.threshold.abs() - D))

        if DGM_c.debug:
            self.A = adj.data.cpu()
            self._x = _x.data.cpu()

        #         self.A=A
        #         A = A/A.sum(-1,keepdim=True)

        if adj.dim() == 2:
            edge_index = adj.nonzero().t()
            edge_attr = adj[edge_index[0], edge_index[1]]
            return edge_index, edge_attr
        else:
            flatten_adj = adj.view(-1, adj.size(-1))
            edge_index = flatten_adj.nonzero().t()
            edges_weight = flatten_adj[edge_index[0], edge_index[1]]

            offset = torch.arange(
                start=0,
                end=adj.size(0) * adj.size(2),
                step=adj.size(2),
                device=adj.device,
            )
            offset = offset.repeat_interleave(adj.size(1))

            edge_index[1] += offset[edge_index[0]]
        edge_index = torch.reshape(edge_index, (2, 1, edge_index.shape[1]))
        edge_index = torch.permute(edge_index,
                                   (1, 0, 2)).squeeze()  # da cambiare in futuro perchè la dimensione deve essere 2
        return x, edge_index, None, edges_weight, None


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

