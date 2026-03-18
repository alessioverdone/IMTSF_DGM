from sklearn.metrics import mean_absolute_error as MAE, root_mean_squared_error, mean_absolute_percentage_error
from torch.nn import ModuleList
from torch_geometric.nn import EdgeConv, GCNConv, GATConv
from torch.nn import functional as F
import torch
import torch.nn as nn
from einops import rearrange
import torch.optim as optim

from src.layers.dgm import DGM_d, MLP, Identity
from src.layers.gcn1d import GCN1DConv, GCN1DConv_big
from src.layers.moe import MixtureOfExpertsDGM


class DGMModel(nn.Module):
    def __init__(self, hparams):
        super(DGMModel, self).__init__()
        self.hparams_ = hparams
        conv_layers = hparams.conv_layers
        dgm_layers = hparams.dgm_layers
        fc_layers = hparams.fc_layers
        self.num_nodes = hparams.num_nodes
        self.dgm_moe = None

        if self.hparams_.dgm_mode == 'simple':
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

            if hparams.pre_fc is not None and len(hparams.pre_fc) > 0:
                self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
            # self.fc = MLP(fc_layers, final_activation=False)
            self.fc = torch.nn.Linear(fc_layers[0], fc_layers[-1])
        elif self.hparams_.dgm_mode == 'moe':
            self.dgm_moe = MixtureOfExpertsDGM(hparams,
                                               hparams.num_nodes,
                                               hparams.node_features,
                                               hparams.prediction_window,
                                               hparams.edge_index,
                                               edge_weight=None,
                                               hidden=hparams.emb_dim)

        self.avg_accuracy = None
        self.debug = False
        self.params = hparams
        self.best_mse = 1000
        self.best_mae = 1000
        self.best_mape = 1000
        self.best_rmse = 1000
        self.optimizer = None  # impostato dopo configure_optimizers()

    def on_validation_epoch_end(self, val_metrics):
        actual_loss = val_metrics['val_mse']
        actual_rmse = val_metrics['val_rmse']
        actual_mae = val_metrics['val_mae']
        actual_mape = val_metrics['val_mape']
        if actual_loss < self.best_mse:
            self.best_mse = actual_loss
            self.best_rmse = actual_rmse
            self.best_mae = actual_mae
            self.best_mape = actual_mape

    def forward(self, x, edges=None, edges_weight=None):
        if self.hparams_.dgm_mode == 'simple':
            if self.params.dataset_name in ['METR-LA', 'solar', 'electricity'] and len(x.shape) == 4:
                x = rearrange(x, 'b t n f -> (b n) (t f)')

            if self.params.pre_fc is not None and len(self.params.pre_fc) > 0:  # [B, N, F]
                x = self.pre_fc(x)
            graph_x = x.detach()
            lprobslist = []
            for f, g in zip(self.graph_f, self.node_g):
                graph_x, edges, lprobs, edges_weight = f(graph_x, edges, None, None)
                graph_x = graph_x.squeeze()
                x_ = torch.reshape(x, (int(x.shape[0] / self.params.num_nodes), self.params.num_nodes, -1))
                b, n, d = x_.shape
                self.edges = edges
                x_pre = g(torch.dropout(x.view(-1, d), self.params.dropout, train=self.training), edges)
                x = torch.nn.functional.relu(x_pre)  # .view(b, n, -1)
                graph_x = torch.cat([graph_x, x.detach()], -1)
                if lprobs is not None:
                    lprobslist.append(lprobs)
            x = F.dropout(x, p=self.params.dropout, training=self.training)
            out = self.fc(x)
            lprobs_out = torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None
        elif self.hparams_.dgm_mode == 'moe':
            out, lprobs_out = self.dgm_moe(x, edges, edges_weight)
        else:
            raise ValueError('dgm_mode must be either "simple" or "moe"')

        return out, lprobs_out[0]

    def configure_optimizers(self):
        if self.hparams_ == 'moe':
            optimizer = torch.optim.Adam(self.dgm_moe.parameters(), lr=self.params.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.params.lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.8)

        return optimizer, scheduler

    def training_step(self, train_batch, batch_idx):
        self.optimizer.zero_grad()

        x, y, edge_index, edge_weight = (train_batch.x,
                                         train_batch.y[:, :self.params.prediction_window],
                                         train_batch.edge_index,
                                         train_batch.edge_attr)
        if self.params.dataset_name in ['METR-LA', 'solar', 'electricity'] and len(x.shape) == 4:
            y = rearrange(y, 'b t n f -> (b n) (t f)')

        y_predicted, logprobs = self.forward(x, edge_index, edge_weight)
        loss_forecasting = F.mse_loss(y_predicted, y)
        loss_forecasting.backward()
        train_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        train_rmse = root_mean_squared_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        train_mape = mean_absolute_percentage_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())

        # GRAPH LOSS (stessa logica del training_step Lightning)
        graph_loss_value = 0.0
        if logprobs is not None:
            corr_pred_ = F.mse_loss(y_predicted, y, reduction="none").detach()
            corr_pred = torch.reshape(corr_pred_, (-1, self.num_nodes, corr_pred_.shape[1]))
            # corr_pred = torch.sum(corr_pred, dim=0)
            # corr_pred = torch.sum(corr_pred, dim=1)  # provato ad usare mean invece di sum che point_w andava ad inf
            corr_pred = torch.mean(corr_pred, dim=0)
            corr_pred = torch.mean(corr_pred, dim=1)
            corr_pred = torch.unsqueeze(corr_pred, dim=0)

            if self.avg_accuracy is None:
                self.avg_accuracy = torch.ones_like(corr_pred) * F.mse_loss(y_predicted, y).detach()

            delta = self.avg_accuracy - corr_pred
            delta = torch.clamp(delta, min=-1, max=1)
            alpha = 0.1
            point_w = delta**2 * torch.exp(-alpha * delta)
            logprobs_ = logprobs.exp().mean([0, -1, -2])
            graph_loss = point_w.squeeze(0) * logprobs_
            graph_loss = graph_loss.mean()
            graph_loss.backward()

            # aggiorna moving average
            self.avg_accuracy = (self.avg_accuracy.to(corr_pred.device) * 0.95 + 0.05 * corr_pred)
            graph_loss_val = graph_loss.detach().item()

        self.optimizer.step()

        current_lr = self.optimizer.param_groups[0]['lr']
        metrics = {
            'train_mse': loss_forecasting.detach().cpu().item(),
            'train_rmse': float(train_rmse),
            'train_mae': float(train_mae),
            'train_mape': float(train_mape),
            'learning_rate': current_lr,
        }
        if graph_loss_val is not None:
            metrics['train_graph_loss'] = graph_loss_val

        return metrics

    def validation_step(self, val_batch, batch_idx):
        x, y, edge_index, edge_weight = (val_batch.x,
                                         val_batch.y[:, :self.params.prediction_window],
                                         val_batch.edge_index,
                                         val_batch.edge_attr)
        if self.params.dataset_name in ['METR-LA', 'solar', 'electricity'] and len(x.shape) == 4:
            y = rearrange(y, 'b t n f -> (b n) (t f)')

        y_predicted, logprobs = self.forward(x, edge_index, edge_weight)
        loss_forecasting = F.mse_loss(y_predicted, y)
        val_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        val_rmse = root_mean_squared_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        val_mape = mean_absolute_percentage_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())

        return {
            'val_mse': loss_forecasting.detach().cpu().item(),
            'val_rmse': float(val_rmse),
            'val_mae': float(val_mae),
            'val_mape': float(val_mape),
        }

    def test_step(self, test_batch, batch_idx):
        x, y, edge_index, edge_weight = (test_batch.x,
                                         test_batch.y[:, :self.params.prediction_window],
                                         test_batch.edge_index,
                                         test_batch.edge_attr)
        if self.params.dataset_name in ['METR-LA', 'solar', 'electricity'] and len(x.shape) == 4:
            y = rearrange(y, 'b t n f -> (b n) (t f)')

        y_predicted, logprobs = self.forward(x, edge_index, edge_weight)
        loss_forecasting = F.mse_loss(y_predicted, y)
        test_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        test_rmse = root_mean_squared_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        test_mape = mean_absolute_percentage_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())

        return {
            'test_mse': loss_forecasting.detach().cpu().item(),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'test_mape': float(test_mape),
        }