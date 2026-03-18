import torch
from sklearn.metrics import mean_absolute_error as MAE, root_mean_squared_error, mean_absolute_percentage_error
from torch import optim, nn
from torch.nn import functional as F

from src.dataset.evaluation import compute_error
from src.utils.utils import get_model





class Training:
    def __init__(self, hparams):
        super(Training, self).__init__()
        self.args = hparams

        # Import model
        self.model = get_model(hparams)

        # Training params
        self.scheduler = None
        self.avg_accuracy = None
        self.debug = False
        self.optimizer = None
        self.best_mse, self.best_mae, self.best_mape, self.best_rmse = float('inf'), float('inf'), float('inf'), float('inf')

    def forward(self, batch_dict):
        # batch_dict.keys = ['observed_data', 'observed_tp', 'observed_mask', 'data_to_predict', 'tp_to_predict', 'mask_predicted_data']
        # NO PATCH
        #`observed_data.shape = (8,55,41)
        # observed_tp.shape = (8,55)
        # observed_mask.shape = (8,55,41)
        # data_to_predict.shape = (8,50,41)
        # tp_to_predict.shape = (8,50)
        # mask_predicted_data.shape = (8,50,41)
        # dati uniformi perchè con padding, in
        # PATCH
        #`observed_data.shape = (8,1,43,41)
        # observed_tp.shape = (8,1,43,41)
        # observed_mask.shape = (8,1,43,41)
        # data_to_predict.shape = (8,50,41)
        # tp_to_predict.shape = (8,50)
        # mask_predicted_data.shape = (8,50,41)
        # dati uniformi perchè con padding, in
        if 'dgm' in self.args.model:
            if self.args.dgm_mode in ['simple', 'moe']:
                pred_y, lprobs_out = self.model(batch_dict['x'],
                                             batch_dict['edges'],
                                             batch_dict['edges_weight'])
                return {'pred_y': pred_y, 'l_probs': lprobs_out[0]}
            else:
                raise ValueError('dgm_mode must be either "simple" or "moe"')

        elif self.args.model == 'hi-patch':
            pred_y = self.model.forecasting(batch_dict["tp_to_predict"],
                                         batch_dict["observed_data"],
                                         batch_dict["observed_tp"],
                                         batch_dict["observed_mask"])
            return {'pred_y':pred_y}

        elif self.args.model == 'prova':
            return self.model(batch_dict["tp_to_predict"],
                              batch_dict["observed_data"],
                              batch_dict["observed_tp"],
                              batch_dict["observed_mask"])
            # return {'pred_y': pred_y}
        else:
            raise ValueError("model must be either 'hi-patch' or 'dgm'")

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode='min',
                                                              patience=2,
                                                              factor=0.8)

    def training_step(self, train_batch):
        self.optimizer.zero_grad()
        res_forward= self.forward(train_batch)

        # Compute losses and optimize DGM
        if 'dgm' in self.args.model:
            results = self.compute_losses_dgm(res_forward['pred_y'],
                                              train_batch['y'],
                                              res_forward['logprobs'])
            results['loss_forecasting'].backward()
            if results['graph_loss'] is not None:
                results['graph_loss'].backward()

        elif self.args.model == 'hi-patch' or self.args.model == 'prova':
            results = self.compute_metrics_and_losses(train_batch,
                                                      res_forward,
                                                     'train')
            results["train_loss"].backward(retain_graph=False)
            if 'graph_loss' in results.keys():
                results['graph_loss'].backward()

        else:
            raise Exception("model must be either 'hi-patch' or 'dgm'")

        self.optimizer.step()

        # Update lr
        results['learning_rate'] = self.optimizer.param_groups[0]['lr']
        results["train_loss"] = results["train_loss"].detach().cpu().float()
        return results

    # def compute_losses_dgm(self, y_predicted, y, logprobs=None):
    #     # Compute losses
    #     loss_forecasting = F.mse_loss(y_predicted, y)
    #
    #     # DGM Graph Loss
    #     graph_loss = None
    #     if logprobs is not None:
    #         corr_pred_ = F.mse_loss(y_predicted, y, reduction="none").detach()
    #         corr_pred = torch.reshape(corr_pred_, (-1, self.num_nodes, corr_pred_.shape[1]))
    #         corr_pred = torch.mean(corr_pred, dim=0)
    #         corr_pred = torch.mean(corr_pred, dim=1)
    #         corr_pred = torch.unsqueeze(corr_pred, dim=0)
    #
    #         if self.avg_accuracy is None:
    #             self.avg_accuracy = torch.ones_like(corr_pred) * F.mse_loss(y_predicted, y).detach()
    #
    #         delta = self.avg_accuracy - corr_pred
    #         delta = torch.clamp(delta, min=-1, max=1)
    #         alpha = 0.1
    #         point_w = delta ** 2 * torch.exp(-alpha * delta)
    #         logprobs_ = logprobs.exp().mean([0, -1, -2])
    #         graph_loss = point_w.squeeze(0) * logprobs_
    #         graph_loss = graph_loss.mean()
    #
    #         # Update moving average
    #         self.avg_accuracy = (self.avg_accuracy.to(corr_pred.device) * 0.95 + 0.05 * corr_pred)
    #
    #     mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
    #     rmse = root_mean_squared_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
    #     mape = mean_absolute_percentage_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
    #
    #     results = {}
    #     results["loss_forecasting"] = loss_forecasting
    #     results["graph_loss"] = graph_loss
    #     results["mae"] = mae.item()
    #     results["rmse"] = rmse.item()
    #     results["mape"] = mape.item()
    #     results['train_graph_loss'] = graph_loss.detach().item()
    #     return results

    def compute_metrics_and_losses(self, batch, res_forward, set_):
        results = {}

        mse = compute_error(batch["data_to_predict"],
                            res_forward['pred_y'],
                            mask=batch["mask_predicted_data"],
                            func="MSE",
                            reduce="mean")
        rmse = torch.sqrt(mse)
        loss = mse

        # Use MSE as the loss function
        # OLD
        # mae = compute_error(batch_dict["data_to_predict"], pred_y, mask = batch_dict["mask_predicted_data"], func="MAE", reduce="mean")
        with torch.no_grad():  # mae non serve per backprop
            mae = compute_error(batch["data_to_predict"],
                                res_forward['pred_y'],
                                mask=batch["mask_predicted_data"],
                                func="MAE", reduce="mean")

        # DGM part
        if 'l_probs' in res_forward.keys():
            logprobs = res_forward['l_probs']
            corr_pred_ = compute_error(batch["data_to_predict"],
                                       res_forward['pred_y'],
                                       mask=batch["mask_predicted_data"],
                                       func="MSE",
                                       reduce="mean").detach()
            corr_pred = torch.reshape(corr_pred_, (-1, self.args.ndim , corr_pred_.shape[1]))
            corr_pred = torch.mean(corr_pred, dim=0)
            corr_pred = torch.mean(corr_pred, dim=1)
            corr_pred = torch.unsqueeze(corr_pred, dim=0)

            if self.avg_accuracy is None:
                self.avg_accuracy = torch.ones_like(corr_pred) * compute_error(batch["data_to_predict"],
                                                                          res_forward['pred_y'],
                                                                          mask=batch["mask_predicted_data"],
                                                                          func="MSE",
                                                                          reduce="mean").detach()

            delta = self.avg_accuracy - corr_pred
            delta = torch.clamp(delta, min=-1, max=1)
            alpha = 0.1
            point_w = delta ** 2 * torch.exp(-alpha * delta)
            logprobs_ = logprobs.exp().mean([0, -1, -2])
            graph_loss = point_w.squeeze(0) * logprobs_
            graph_loss = graph_loss.mean()
            results[f'{set_}_graph_loss'] = graph_loss

            # Update moving average
            self.avg_accuracy = (self.avg_accuracy.to(corr_pred.device) * 0.95 + 0.05 * corr_pred)

        # Store the loss and error metrics
        results[f'{set_}_loss'] = loss
        results[f'{set_}_mse'] = mse.item()
        results[f'{set_}_rmse'] = rmse.item()
        results[f'{set_}_mae'] = mae.item()
        return results

    def validation_step(self, val_batch):
        res_forward = self.forward(val_batch)

        # Compute metrics
        if 'dgm' in self.args.model:
            results = self.compute_losses_dgm(res_forward['pred_y'],
                                              val_batch['y'],
                                              res_forward['logprobs'])
        elif self.args.model == 'hi-patch' or self.args.model == 'prova':
            results = self.compute_metrics_and_losses(val_batch,
                                                      res_forward,
                                                      'val')
            results["val_loss"] = results["val_loss"].detach().cpu().float()
        else:
            raise Exception("model must be either 'hi-patch' or 'dgm'")
        return results

    def test_step(self, test_batch,):
        res_forward = self.forward(test_batch)

        # Compute losses and optimize DGM
        if 'dgm' in self.args.model:
            results = self.compute_losses_dgm(res_forward['pred_y'],
                                              test_batch['y'],
                                              res_forward['logprobs'])
        elif self.args.model == 'hi-patch' or self.args.model == 'prova':
            results = self.compute_metrics_and_losses(test_batch,
                                                      res_forward,
                                                      'test')
            results["test_loss"] = results["test_loss"].detach().cpu().float()
        else:
            raise Exception("model must be either 'hi-patch' or 'dgm'")
        return results

    def on_validation_epoch_end(self, val_metrics):
        actual_loss = val_metrics['val_mse']
        actual_rmse = val_metrics['val_rmse']
        actual_mae = val_metrics['val_mae']
        # actual_mape = val_metrics['val_mape']
        if actual_loss < self.best_mse:
            self.best_mse = actual_loss
            self.best_rmse = actual_rmse
            self.best_mae = actual_mae
            # self.best_mape = actual_mape