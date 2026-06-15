"""
model_backbone_phitp.py
=======================

TSIL backbone driven by a single ``phi_type`` switch (there is no ``loss_type``
field). ``_select_criterion`` supports the two loss modes:

    phi_type = 'mse'   -> plain MSE (no statistical-invariant weighting)
    phi_type = 'ones'  -> weighted MSE with the all-ones control predicate Phi_ones
    phi_type = <TP>    -> weighted MSE with one of the six temporal predicates
                          (phi_timestamp / phi_cycle / phi_phase /
                           phi_momentum / phi_volatility / phi_return_dist),
                          defined in ``src/temporal_predicates.py``.

The same six predicates also have an alternate implementation in
``src/custom_phi_constructors.py``, served as a fallback when their key is supplied.

This is the single QniverseModel backbone used by ``train.py`` / ``search.py``.
"""

import os
import copy
import json
import collections
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time

from qlib.log import get_module_logger
from qlib.model.base import Model

from types import SimpleNamespace
import matplotlib.pyplot as plt

# vis
import plotly
from qlib.contrib.report.analysis_model.analysis_model_performance import model_performance_graph

from src.models.PatchTST import PatchTST
# The following backbones are not part of the released experiments and their
# source files are omitted from this release. Re-enable the import after adding
# the corresponding file under src/models/ if you need them.
# from src.models.PDF import PDF
# from src.models.TimeMixer import TimeMixer
# from src.models.TimesNet import TimesNet
# from src.models.SegRNN import SegRNN
# from src.models.diffusion_stock import DiffStock
# from src.models.Crossformer import Crossformer
from src.models.LSTM import LSTM
from src.models.GRU import GRU
from src.models.Transformer import Transformer
# from src.models.Mamba import mamba
from src.models.TCN import TCN
from src.models.GAT import GAT
from src.models.GCN import GCN
from src.models.LSR_IGRU import LSR_IGRU, LSRIGRU
from src.models.StockMixer import StockMixer
from src.models.master import MASTER
from src.models.MERA import MERA

# The six paper Temporal Predicates (timestamp / cycle / phase / momentum /
# volatility / return-distribution) live in their own module.
from src.temporal_predicates import get_temporal_I_P, TEMPORAL_PHI_TYPES

# Alternate implementation of the paper's predicates in src/custom_phi_constructors.py.
# Optional: if the module is present, these keys can also be served from it. The
# whitelist below is the paper's set only: the six temporal predicates plus the
# Phi_ones control.
try:
    from src.custom_phi_constructors import get_custom_I_P
    _CUSTOM_PHI_TYPES = {
        'phi_timestamp', 'phi_cycle', 'phi_phase',
        'phi_momentum', 'phi_volatility', 'phi_return_dist',
        'ones',
    }
    _CUSTOM_PHI_AVAILABLE = True
except ImportError:
    _CUSTOM_PHI_AVAILABLE = False
    _CUSTOM_PHI_TYPES = set()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_ones_I_P(self, batch_size, device):
        """Phi_ones control predicate: build the I and P matrices with phi = 1."""
        I = torch.eye(batch_size, device=device)  # identity matrix (batch_size x batch_size)
        phi = torch.ones((batch_size, 1), device=device)
        phi = phi / torch.linalg.norm(phi)  # unit-norm normalization
        P = torch.mm(phi, phi.T)
        return I, P

    def forward(self, phi_type, batch_x, timestamp_x, outputs, targets, tau_hat=None, tau=None):
        batch_size = outputs.shape[0]
        device = outputs.device
        if phi_type in TEMPORAL_PHI_TYPES:
            # One of the paper's six temporal predicates
            # (timestamp / cycle / phase / momentum / volatility / return-distribution).
            I, P = get_temporal_I_P(phi_type, batch_x, timestamp_x=timestamp_x, device=device)
        elif _CUSTOM_PHI_AVAILABLE and phi_type in _CUSTOM_PHI_TYPES:
            # Same predicates served from src/custom_phi_constructors.py.
            I, P = get_custom_I_P(phi_type, batch_x, device=device)
        else:
            # 'ones' control predicate Φ_ones (also the fallback for any unknown key).
            I, P = self.get_ones_I_P(batch_size, device)
        # LUSI loss is the QUADRATIC form (Vapnik & Izmailov 2020, eq. 79):
        #   R = (out - tgt)^T (tau_hat * V + tau * P) (out - tgt),  here V = I.
        # It must be built on the RAW residual e = (out - tgt), NOT on e**2.
        # The P term  tau * e^T P e = tau * ||Phi^T e||^2  is the statistical
        # invariant; it contains the cross products e_i * e_j. Squaring e first
        # With the form below
        # V_loss + P_loss == total exactly (M is linear in the e^T M e form).
        resid = (outputs - targets).reshape(-1, 1)  # e, column vector [bs, 1]
        if tau_hat is not None:
            M = tau_hat * I + tau * P                        # τ̂V + τP, here V=I
            Me = torch.matmul(M, resid)                      # (τ̂I+τP) e
            total = torch.mean(resid * Me)                   # (1/n) eᵀ M e
            return {
                'total': total,
                'ori_mse': torch.mean(resid.pow(2)),
                'V_loss': torch.mean(resid * torch.matmul(tau_hat * I, resid)),  # τ̂ eᵀI e
                'P_loss': torch.mean(resid * torch.matmul(tau * P, resid)),      # τ eᵀP e
            }
        else:
            assert tau_hat is None, "loss type error..."

class RankMSELoss(nn.Module):
    def __init__(self, rank_weight=3.0, mse_weight=1.0):
        super(RankMSELoss, self).__init__()
        self.rank_weight = rank_weight
        self.mse_weight = mse_weight

    def forward(self, pred, label):
        mse_loss = (pred - label).pow(2).mean()
        # rank predictions vs. ground truth
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0) # pairwise differences
        label_diff = label.unsqueeze(1) - label.unsqueeze(0)
    
        rank_loss = F.relu(- (pred_diff * label_diff)) # same sign => correct order => no penalty
        
        rank_loss = rank_loss.sum(dim=[1])
        rank_loss = rank_loss.mean()

        combined_loss = self.rank_weight * rank_loss + self.mse_weight * mse_loss
        
        return combined_loss
    

class QniverseModel(Model):
    def __init__(
            self,
            model_config,
            model_type="WFTNet",
            lr=1e-3,
            n_epochs=500,
            early_stop=50,
            smooth_steps=5,
            max_steps_per_epoch=None,
            freeze_model=False,
            model_init_state=None,
            seed=None,
            logdir=None,
            eval_train=True,
            eval_test=False,
            avg_params=True,
            plot_training=True,  # whether to plot the training curves
            plot_show=False,
            **kwargs,
    ):

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.logger = get_module_logger("Qniverse")
        self.logger.info("Qniverse Model...")

        self.model_type = model_type
        self.model = eval(model_type)(SimpleNamespace(**model_config)).to(device)
        if model_init_state:
            self.model.load_state_dict(torch.load(model_init_state, map_location="cpu")["model"])
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad_(False)
        else:
            self.logger.info("# model params: %d" % sum([p.numel() for p in self.model.parameters()]))

        self.optimizer = optim.Adam(list(self.model.parameters()), lr=lr)

        self.model_config = model_config
        self.lr = lr
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.smooth_steps = smooth_steps
        self.max_steps_per_epoch = max_steps_per_epoch
        self.seed = seed
        self.logdir = logdir
        self.eval_train = eval_train
        self.eval_test = eval_test
        self.avg_params = avg_params
        # loss_type is removed: the loss mode is driven solely by phi_type.
        #   phi_type == 'mse'  -> plain MSE
        #   phi_type != 'mse'  -> weighted MSE with P built from the predicate
        #                         ('ones' = Φ_ones control, or a temporal predicate key)
        self.phi_type = model_config.get('phi_type', 'mse')
        self.use_weighted = (self.phi_type != 'mse')
        self.plot_training = plot_training  # plotting flag
        self.plot_show = plot_show

        self.criterion = self._select_criterion()

        if self.logdir is not None:
            if os.path.exists(self.logdir):
                self.logger.warn(f"logdir {self.logdir} is not empty")
            os.makedirs(self.logdir, exist_ok=True)

        self.global_step = -1

    def _select_criterion(self):
        """Pick the loss based solely on phi_type.

        phi_type == 'mse' -> plain MSE (RankMSELoss with the ranking term disabled).
        otherwise         -> WeightedMSELoss; the predicate (phi_type) decides P:
                             'ones' = Φ_ones control, or one of the temporal predicates.
        """
        if self.phi_type == 'mse':
            criterion = RankMSELoss(rank_weight=0.0, mse_weight=1.0)
        else:
            criterion = WeightedMSELoss()
        return criterion


    def plot_training_curves(self, train_logs, valid_logs, best_epoch, show=False):
        """Plot and save the training curves."""
        train_logs = pd.DataFrame(train_logs).reset_index()
        valid_logs = pd.DataFrame(valid_logs).reset_index()
        epochs = range(1, len(train_logs) + 1)

        # 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # MSE
        ax1.plot(epochs, train_logs['MSE'], 'r-', label='Training')
        ax1.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax1.set_title('mse Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('train mse Loss')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax1_right = ax1.twinx()
        ax1_right.plot(epochs, valid_logs['MSE'], 'g-', label='Validation')
        ax1_right.set_ylabel('val mse Loss')#, color='purple'
        ax1_right.tick_params(axis='y')#, labelcolor='purple'
        ax1_right.legend(loc='upper right')

        # MAE
        ax2.plot(epochs, train_logs['MAE'], 'r-', label='Training')
        ax2.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax2.set_title('mae Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('train mae Loss')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        ax2_right = ax2.twinx()
        ax2_right.plot(epochs, valid_logs['MAE'], 'g-', label='Validation')
        ax2_right.set_ylabel('val mae Loss')#, color='purple'
        ax2_right.tick_params(axis='y')#, labelcolor='purple'
        ax2_right.legend(loc='upper right')

        # Correlation
        ax3.plot(epochs, train_logs['IC'], 'r-', label='Training')
        ax3.plot(epochs, valid_logs['IC'], 'g-', label='Validation')
        ax3.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax3.set_title('IC')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('IC')
        ax3.legend()
        ax3.grid(True)

        # Rank IC
        ax4.plot(epochs, train_logs['Rank_IC'], 'r-', label='Training')
        ax4.plot(epochs, valid_logs['Rank_IC'], 'g-', label='Validation')
        ax4.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax4.set_title('Rank IC')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Rank IC')
        ax4.legend()
        ax4.grid(True)

        if show==True:
            plt.show()
        plt.tight_layout()
        fig_name = 'training_curves_tau_'+str(self.model_config['ind'])+'.png'
        plt.savefig(os.path.join(self.logdir, fig_name), dpi=300, bbox_inches='tight')
        plt.close()

        # also save the raw data as a CSV file
        training_data = pd.DataFrame({
            'epoch': epochs,
            'train_ic': train_logs['IC'],
            'train_rank_ic': train_logs['Rank_IC'],
            'train_mse': train_logs['MSE'],
            'train_mae': train_logs['MAE'],
            'valid_ic': valid_logs['IC'],
            'valid_rank_ic': valid_logs['Rank_IC'],
            'valid_mse': valid_logs['MSE'],
            'valid_mae': valid_logs['MAE']
        })
        csv_file_name = 'training_data_tau_'+str(self.model_config['tau_hat_init'])+'.csv'
        training_data.to_csv(os.path.join(self.logdir, csv_file_name), index=False)


    def plot_training_curves_tau(self, train_logs, valid_logs, best_epoch, show=False):
        """Plot and save the training curves."""
        train_logs = pd.DataFrame(train_logs).reset_index()
        valid_logs = pd.DataFrame(valid_logs).reset_index()
        epochs = range(1, len(train_logs) + 1)

        # 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # loss
        ax1.tick_params(axis='y')
        ax1.plot(epochs, train_logs['Weight_loss'], 'r-', label='Total')
        ax1.plot(epochs, train_logs['V_loss'], 'g-', label='V_loss')
        # ax1.plot(epochs, train_logs['P_loss'], 'b-', label='P_loss')
        ax1.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax1.set_title('Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('total&V loss')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax1_right = ax1.twinx()
        ax1_right.plot(epochs, train_logs['P_loss'], 'b-', label='P_loss')
        ax1_right.set_ylabel('P loss')#, color='purple'
        ax1_right.tick_params(axis='y')#, labelcolor='purple'
        ax1_right.legend(loc='upper right')

        # MSE
        ax2.tick_params(axis='y')#, labelcolor='orange'
        ax2.plot(epochs, train_logs['MSE'], 'r-', label='Train_mse')
        ax2.plot(epochs, train_logs['MAE'], 'b-', label='Train_mae')
        # ax2.plot(epochs, valid_logs['MSE'], 'g-', label='Val_mse')
        # ax2.plot(epochs, valid_logs['MAE'], 'o-', label='Val_mae')
        ax2.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax2.set_title('MSE/MAE')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('train loss')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        ax2_right = ax2.twinx()
        ax2_right.plot(epochs, valid_logs['MSE'], 'go-', label='Val_mse')
        ax2_right.plot(epochs, valid_logs['MAE'], 'mo-', label='Val_mae')
        ax2_right.set_ylabel('val loss')#, color='purple'
        ax2_right.tick_params(axis='y')#, labelcolor='purple'
        ax2_right.legend(loc='upper right')

        # Correlation
        ax3.plot(epochs, train_logs['Tau_P'], 'r-', label='Tau_P')
        ax3.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax3.set_title('Tau_P')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Tau_P')
        ax3.legend()
        ax3.grid(True)

        # Rank IC
        ax4.plot(epochs, train_logs['IC'], 'r-', label='Train_IC')
        ax4.plot(epochs, valid_logs['IC'], 'go-', label='Val_IC')
        ax4.plot(epochs, train_logs['Rank_IC'], 'b-', label='Train_RIC')
        ax4.plot(epochs, valid_logs['Rank_IC'], 'mo-', label='Val_RIC')
        ax4.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax4.set_title('Correlation')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Correlation')
        ax4.legend()
        ax4.grid(True)

        if show==True:
            plt.show()
        plt.tight_layout()
        fig_name = 'training_curves_tau_'+str(self.model_config['ind'])+'.png'
        plt.savefig(os.path.join(self.logdir, fig_name), dpi=300, bbox_inches='tight')
        plt.close()

        # also save the raw data as a CSV file
        training_data = pd.DataFrame({
            'epoch': epochs,
            'tau_P': train_logs['Tau_P'],
            'total_loss': train_logs['Weight_loss'],
            'v_loss': train_logs['V_loss'],
            'p_loss': train_logs['P_loss'],
            'train_rank_ic': train_logs['Rank_IC'],
            'train_mse': train_logs['MSE'],
            'train_mae': train_logs['MAE'],
            'valid_ic': valid_logs['IC'],
            'valid_rank_ic': valid_logs['Rank_IC'],
            'valid_mse': valid_logs['MSE'],
            'valid_mae': valid_logs['MAE']
        })
        csv_file_name = 'training_data_tau_'+str(self.model_config['tau_hat_init'])+'.csv'
        training_data.to_csv(os.path.join(self.logdir, csv_file_name), index=False)


    def train_epoch(self, data_set):

        self.model.train()

        data_set.train()

        max_steps = len(data_set)
        if self.max_steps_per_epoch is not None:
            max_steps = min(self.max_steps_per_epoch, max_steps)

        count = 0
        total_loss = 0
        total_count = 0
        v_loss = 0
        p_loss = 0

        metrics = []

        for batch in data_set: # add
            count += 1
            if count > max_steps:
                break

            self.global_step += 1

            data, label, index = batch["data"], batch["label"], batch["index"]
            timestamp = batch.get("timestamp_feature")  # None unless phi_type is a timestamp predicate
            data, label, index = data.to(device), label.to(device), index.to(device)
            if timestamp is not None:
                timestamp = timestamp.to(device)
            feature = data[:, :, : -1]

            feature = torch.nan_to_num(feature, nan=0.0)
            label = torch.nan_to_num(label, nan=0.0)
            if timestamp is not None:
                timestamp = torch.nan_to_num(timestamp, nan=0.0)

            # feature [batch_size, seq_len, num_fea]
            pred = self.model(feature).squeeze() # B
            
            pred = torch.nan_to_num(pred, nan=0.0)

            # pred [batch_size, horizon]
            if self.use_weighted:
                tau_hat = torch.sigmoid(self.model.alpha)
                tau = 1 - tau_hat
                loss_dict = self.criterion(self.phi_type, feature, timestamp, pred, label, tau_hat, tau)
                loss = loss_dict['total']
            else:
                loss = self.criterion(pred, label)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            total_count += len(pred)

            if self.use_weighted:
                v_loss += loss_dict['V_loss'].item()
                p_loss += loss_dict['P_loss'].item()

            # compute IC and related metrics
            X = np.c_[
                pred.detach().cpu().numpy(),
                label.cpu().numpy(),
            ]
            columns = ["score", "label"]
            pred = pd.DataFrame(X, index=index.cpu().numpy(), columns=columns)
            metrics.append(evaluate(pred))

        total_loss /= total_count

        if self.use_weighted:
            metrics = pd.DataFrame(metrics)
            metrics = {
                "MSE": metrics.MSE.mean(),
                "MAE": metrics.MAE.mean(),
                "IC": metrics.IC.mean(),
                "Rank_IC": metrics.Rank_IC.mean(),
                'Weight_loss': total_loss,
                'V_loss': v_loss/total_count,
                'P_loss': p_loss/total_count,
                'Tau_P': torch.sigmoid(self.model.alpha).item(),  # weight of the P matrix over training
            }
        else:
            metrics = pd.DataFrame(metrics)
            metrics = {
                "MSE": metrics.MSE.mean(),
                "MAE": metrics.MAE.mean(),
                "IC": metrics.IC.mean(),
                "Rank_IC": metrics.Rank_IC.mean(),
            }

        return total_loss, metrics

    def test_epoch(self, data_set, return_pred=False):

        self.model.eval()
        data_set.eval()

        preds = []
        metrics = []
        total_inference_time = 0.0

        for batch in data_set:
            data, label, index = batch["data"], batch["label"], batch["index"]

            feature = data[:, :, : -1]

            feature = torch.nan_to_num(feature, nan=0.0)
            label = torch.nan_to_num(label, nan=0.0)
            with torch.no_grad():
                start_time = time.time()
                pred = self.model(feature).squeeze()
                end_time = time.time()
                pred = torch.nan_to_num(pred, nan=0.0)

            total_inference_time += end_time - start_time

            X = np.c_[
                pred.cpu().numpy(),
                label.cpu().numpy(),
            ]
            columns = ["score", "label"]

            pred = pd.DataFrame(X, index=index.cpu().numpy(), columns=columns)

            metrics.append(evaluate(pred))

            if return_pred:
                preds.append(pred)

        metrics = pd.DataFrame(metrics)
        metrics = {
            "InfT": total_inference_time / len(data_set),
            "MSE": metrics.MSE.mean(),
            "MAE": metrics.MAE.mean(),
            "IC": metrics.IC.mean(),
            "Rank_IC": metrics.Rank_IC.mean(),
        }


        if return_pred:
            preds = pd.concat(preds, axis=0)
            preds.index = data_set.restore_index(preds.index) # 
            preds.index = preds.index.swaplevel()
            preds.sort_index(inplace=True)
        print("preds")
        print(preds)
        return metrics, preds
    
    def fit(self, dataset, evals_result=dict()):
        
        train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])

        best_score = -1
        best_epoch = 0
        stop_rounds = 0
        best_params = {
            "model": copy.deepcopy(self.model.state_dict()),
        }
        params_list = {
            "model": collections.deque(maxlen=self.smooth_steps),
        }
        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["test"] = []

        # train
        self.global_step = -1

        train_logs = []  # each element is a dict of metrics
        valid_logs = []

        for epoch in range(self.n_epochs):
            self.logger.info("Epoch %d:", epoch)

            self.logger.info("training...")
            _, train_metrics = self.train_epoch(train_set)

            self.logger.info("evaluating...")
            # average params for inference
            params_list["model"].append(copy.deepcopy(self.model.state_dict()))
            self.model.load_state_dict(average_params(params_list["model"]))

            valid_metrics = self.test_epoch(valid_set)[0]
            evals_result["valid"].append(valid_metrics)
            self.logger.info("\tvalid metrics: %s" % valid_metrics)

            if self.eval_test:
                test_metrics = self.test_epoch(test_set)[0]
                evals_result["test"].append(test_metrics)
                self.logger.info("\ttest metrics: %s" % test_metrics)

            if valid_metrics["IC"] > best_score:
                self.logger.info("\tvalid ic increased: %s" % (valid_metrics["IC"]-best_score))
                best_score = valid_metrics["IC"]
                stop_rounds = 0
                best_epoch = epoch
                best_params = {
                    "model": copy.deepcopy(self.model.state_dict()),
                }
            else:
                stop_rounds += 1
                if stop_rounds >= self.early_stop:
                    self.logger.info("early stop @ %s" % epoch)
                    break

            if self.use_weighted:
                train_logs.append({
                    'IC': train_metrics['IC'],
                    'Rank_IC': train_metrics['Rank_IC'],
                    'Weight_loss': train_metrics['Weight_loss'],
                    'V_loss': train_metrics['V_loss'],
                    'P_loss': train_metrics['P_loss'],
                    'MSE': train_metrics['MSE'],
                    'MAE': train_metrics['MAE'],
                    'Tau_P': train_metrics['Tau_P']
                })

                valid_logs.append({
                    'IC': valid_metrics['IC'],
                    'Rank_IC': valid_metrics['Rank_IC'],
                    'MSE': valid_metrics['MSE'],
                    'MAE': valid_metrics['MAE']
                })
            else:
                train_logs.append({
                    'IC': train_metrics['IC'],
                    'Rank_IC': train_metrics['Rank_IC'],
                    'MSE': train_metrics['MSE'],
                    'MAE': train_metrics['MAE']
                })

                valid_logs.append({
                    'IC': valid_metrics['IC'],
                    'Rank_IC': valid_metrics['Rank_IC'],
                    'MSE': valid_metrics['MSE'],
                    'MAE': valid_metrics['MAE']
                })

            # restore parameters
            self.model.load_state_dict(params_list["model"][-1])

        # plot the training curves after training
        if self.plot_training and self.logdir:
            if self.use_weighted:
                self.plot_training_curves_tau(train_logs, valid_logs, best_epoch, self.plot_show)
            else:
                self.plot_training_curves(train_logs, valid_logs, best_epoch, self.plot_show)

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_params["model"])

        metrics, preds = self.test_epoch(test_set, return_pred=True)
        metrics['best_epoch'] = best_epoch
        metrics['best_score'] = best_score
        self.logger.info("test metrics: %s" % metrics)

        if self.logdir:
            self.logger.info("save model & pred to local directory")

            torch.save(best_params, self.logdir + "/model.bin")

            fig_list = model_performance_graph(preds, show_notebook=False)
            fig_name = [f"{self.model_type}_cumulative_return", f"{self.model_type}_distribution_return", f"{self.model_type}_IC",
                        f"{self.model_type}_monthly_IC", f"{self.model_type}_distribution_IC", f"{self.model_type}_auto_corr"]
            for i, fig in enumerate(fig_list):
                fig: plotly.graph_objs.Figure = fig
                fig.write_html(self.logdir + f'/{fig_name[i]}.html')
            print("Vis Finished!")

            preds.to_pickle(self.logdir + "/pred.pkl")

            metrics = {k: float(v) if isinstance(v, np.floating) else v for k, v in metrics.items()}
            self.model_config['tau_hat_init'] = float(self.model_config['tau_hat_init'])
            info = {
                "config": {
                    "model_config": self.model_config,
                    "lr": self.lr,
                    "n_epochs": self.n_epochs,
                    "early_stop": self.early_stop,
                    "smooth_steps": self.smooth_steps,
                    "max_steps_per_epoch": self.max_steps_per_epoch,
                    "seed": self.seed,
                    "logdir": self.logdir,
                },
                "best_eval_metric": -best_score,  # NOTE: minux -1 for minimize
                "metric": metrics,
            }
            with open(self.logdir + "/info.json", "w") as f:
                json.dump(info, f)

            return preds, metrics

    def predict(self, dataset, segment="test"):

        test_set = dataset.prepare(segment)

        metrics, preds = self.test_epoch(test_set, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)

        return preds

def evaluate_ori(pred):
    pred = pred.rank(pct=True)  # transform into percentiles
    score = pred.score
    label = pred.label
    diff = score - label
    MSE = (diff ** 2).mean()
    MAE = (diff.abs()).mean()
    IC = score.corr(label, method="pearson")
    Rank_IC = score.corr(label, method="spearman")
    return {"MSE": MSE, "MAE": MAE, "IC": IC,  "Rank_IC": Rank_IC}

def evaluate(pred):
    score = pred.score
    label = pred.label
    diff = score - label
    MSE = (diff ** 2).mean()
    MAE = (diff.abs()).mean()
    # Pearson IC
    IC = score.corr(label, method="pearson")
    # Spearman rank IC
    Rank_IC = score.corr(label, method="spearman")
    return {"MSE": MSE, "MAE": MAE, "IC": IC, "Rank_IC": Rank_IC}

def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError("the %d-th model has different params" % i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def sinkhorn(Q, n_iters=3, epsilon=0.01):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q
