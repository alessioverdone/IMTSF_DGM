from einops import rearrange
from sklearn.metrics import mean_absolute_error as MAE, root_mean_squared_error, mean_absolute_percentage_error
from torch import optim, nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import functional as F

from src.layers.dgm import MLP
from src.layers.gcn1d import GCN1DConv_big, GCN1DConv
from src.layers.moe import MixtureOfExperts


class BaselineModelPV(nn.Module):
    def __init__(self, params):
        super(BaselineModelPV, self).__init__()
        hidden_channels = params.emb_dim
        node_features = params.node_features
        if params.model == 'gcn':
            self.conv1 = GCNConv(node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
        elif params.model == 'gat':
            self.conv1 = GATConv(node_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
            self.conv3 = GATConv(hidden_channels, hidden_channels)
        elif params.model == 'mlp':
            self.conv1 = nn.Linear(node_features, params.dgm_layers[0])
            self.conv2 = MLP(params.dgm_layers)
            self.conv3 = MLP(params.dgm_layers)
        elif params.model == 'gcn1d':
            self.gcn1d_model = GCN1DConv(node_features, params.prediction_window, 1, 1, hid_dim=hidden_channels)
        elif params.model == 'gcn1d-big':
            self.gcn1d_big_model = GCN1DConv_big(node_features, params.prediction_window, 1, 1, hid_dim=hidden_channels)
        elif params.model == 'moe':
            self.moe = MixtureOfExperts(params.num_nodes,
                                        node_features,
                                        params.prediction_window,
                                        params.edge_index,
                                        edge_weight=None,
                                        hidden=hidden_channels)
        self.lin = Linear(hidden_channels, params.prediction_window)
        self.params = params
        self.debug = False
        self.best_mse = 1000
        self.best_mae = 1000
        self.best_rmse = 1000
        self.best_mape = 1000
        self.optimizer = None  # impostato dopo configure_optimizers()

    def forward(self, data):
        x, y, edge_index, edge_weight = (data.x,
                                         data.y[:, :self.params.prediction_window],
                                         data.edge_index,
                                         data.edge_attr)
        if ((self.params.dataset_name == 'METR-LA' or self.params.dataset_name == 'solar')
                and len(x.shape) == 4
                and self.params.model != 'moe'):
            x = rearrange(x, 'b t n f -> (b n) (t f)')

        if self.params.model in ['gcn', 'gat']:
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)
            x = F.dropout(x, p=self.params.dropout, training=self.training)
            x = self.lin(x)
        elif self.params.model == 'gcn1d':
            x = self.gcn1d_model(x, edge_index)
        elif self.params.model == 'gcn1d-big':
            x = self.gcn1d_big_model(x, edge_index)
        elif self.params.model == 'moe':
            x = self.moe(x)
        return x

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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.params.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.15)
        return optimizer, scheduler

    def training_step(self, train_batch, batch_idx):
        self.optimizer.zero_grad()

        x, y, edge_index, edge_weight = (train_batch.x,
                                         train_batch.y[:, :self.params.prediction_window],
                                         train_batch.edge_index,
                                         train_batch.edge_attr)
        if self.params.dataset_name in ['METR-LA', 'solar', 'electricity'] and len(x.shape) == 4:
            y = rearrange(y, 'b t n f -> (b n) (t f)')

        y_predicted = self.forward(train_batch)
        loss_forecasting = F.mse_loss(y_predicted, y)
        loss_forecasting.backward()
        train_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        train_rmse = root_mean_squared_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        train_mape = mean_absolute_percentage_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        self.optimizer.step()

        return {
            'train_mse': loss_forecasting.detach().cpu().item(),
            'train_rmse': float(train_rmse),
            'train_mae': float(train_mae),
            'train_mape': float(train_mape),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }

    def validation_step(self, val_batch, batch_idx):
        x, y, edge_index, edge_weight = (val_batch.x,
                                         val_batch.y[:, :self.params.prediction_window],
                                         val_batch.edge_index,
                                         val_batch.edge_attr)
        if self.params.dataset_name in ['METR-LA', 'solar', 'electricity'] and len(x.shape) == 4:
            y = rearrange(y, 'b t n f -> (b n) (t f)')

        y_predicted = self.forward(val_batch)
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

        y_predicted = self.forward(test_batch)
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
