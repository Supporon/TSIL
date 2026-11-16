"""Nine released backbones with paper-faithful, training-only TSIL penalties.

Original uses MSE; TSIL adds fixed-weight residual moments; Feature concatenates
the same raw descriptors at every input time step. Validation is daily IC.
"""

import os
import copy
import json
import collections
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import time

from qlib.log import get_module_logger
from qlib.model.base import Model

from types import SimpleNamespace
import matplotlib.pyplot as plt

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
from src.models.LSR_IGRU import LSR_IGRU, LSRIGRU
from src.models.StockMixer import StockMixer
from src.models.master import MASTER
from src.models.MERA import MERA

from src.temporal_predicates import selected_predicates, descriptor_dimensions
from src.tsil_loss import StatisticalInvariantLoss
from src.run_utils import RELATIONAL_MODELS

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class QniverseModel(Model):
    def __init__(
            self,
            model_config,
            model_type="WFTNet",
            lr=1e-3,
            n_epochs=500,
            early_stop=20,
            smooth_steps=1,
            max_steps_per_epoch=None,
            freeze_model=False,
            model_init_state=None,
            seed=None,
            logdir=None,
            eval_train=True,
            eval_test=False,
            avg_params=False,
            plot_training=True,  # whether to plot the training curves
            plot_show=False,
            **kwargs,
    ):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.logger = get_module_logger("Qniverse")
        self.logger.info("Qniverse Model...")

        self.model_type = model_type
        self.model_config = copy.deepcopy(model_config)
        self.phi_type = model_config.get("phi_type", "mse")
        self.predicates = selected_predicates(self.phi_type)
        self.injection_mode = model_config.get("injection_mode", "tsil" if self.predicates else "original")
        if self.injection_mode not in {"original", "tsil", "feature"}:
            raise ValueError("injection_mode must be original, tsil, or feature")
        if self.injection_mode == "feature" and not self.predicates:
            raise ValueError("feature injection requires at least one predicate")
        alpha = model_config.get("penalty_alpha", 0.0)
        alphas = alpha if isinstance(alpha, dict) else {name: alpha for name in self.predicates}
        if any(not math.isfinite(float(alphas[name])) for name in self.predicates):
            raise ValueError("penalty_alpha must be finite and fixed within a run")
        self.betas = {name: torch.sigmoid(torch.tensor(float(alphas[name]), dtype=torch.float64)).item()
                      for name in self.predicates}
        self.use_weighted = self.injection_mode == "tsil" and bool(self.predicates)
        backbone_config = copy.deepcopy(model_config)
        if self.injection_mode == "feature":
            dims = descriptor_dimensions(model_config["seq_len"], model_config.get("predicate_config"))
            backbone_config["enc_in"] += sum(dims[name] for name in self.predicates)
        self.model = eval(model_type)(SimpleNamespace(**backbone_config)).to(device)
        if model_init_state:
            self.model.load_state_dict(torch.load(model_init_state, map_location="cpu")["model"])
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad_(False)
        else:
            self.logger.info("# model params: %d" % sum([p.numel() for p in self.model.parameters()]))

        self.optimizer = optim.Adam(list(self.model.parameters()), lr=lr)

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
        self.plot_training = plot_training  # plotting flag
        self.plot_show = plot_show

        self.criterion = self._select_criterion()

        if self.logdir is not None:
            if os.path.exists(self.logdir):
                self.logger.warn(f"logdir {self.logdir} is not empty")
            os.makedirs(self.logdir, exist_ok=True)

        self.global_step = -1

    def _select_criterion(self):
        return StatisticalInvariantLoss(self.betas) if self.use_weighted else nn.MSELoss()

    def _forward_batch(self, batch):
        feature = batch["data"].to(device)[:, :, :-1]
        feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
        if self.injection_mode == "feature":
            descriptors = batch.get("predicates", {})
            if set(descriptors) != set(self.predicates):
                raise ValueError("batch descriptors do not match the selected feature predicates")
            extra = torch.cat([descriptors[name].to(feature) for name in self.predicates], dim=1)
            feature = torch.cat([feature, extra[:, None, :].expand(-1, feature.shape[1], -1)], dim=2)
        if self.model_type not in RELATIONAL_MODELS:
            return self.model(feature).reshape(-1)
        # Relationship models see a complete cross-section from ONE date only.
        # The loss may subsequently join multiple dates into a stock-time batch.
        dates = batch["date"].to(device)
        result = feature.new_empty(feature.shape[0])
        for date in torch.unique(dates):
            mask = dates == date
            result[mask] = self.model(feature[mask]).reshape(-1)
        return result


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
        fig_name = 'training_curves_tau_'+str(self.model_config.get('ind', 0))+'.png'
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
        csv_file_name = 'training_data_tau_'+str(self.model_config.get('penalty_alpha', 0.0))+'.csv'
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
        ax3.plot(epochs, train_logs['Beta'], 'r-', label='Beta')
        ax3.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax3.set_title('Beta')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Beta')
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
        fig_name = 'training_curves_tau_'+str(self.model_config.get('ind', 0))+'.png'
        plt.savefig(os.path.join(self.logdir, fig_name), dpi=300, bbox_inches='tight')
        plt.close()

        # also save the raw data as a CSV file
        training_data = pd.DataFrame({
            'epoch': epochs,
            'beta': train_logs['Beta'],
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
        csv_file_name = 'training_data_tau_'+str(self.model_config.get('penalty_alpha', 0.0))+'.csv'
        training_data.to_csv(os.path.join(self.logdir, csv_file_name), index=False)


    def train_epoch(self, data_set):
        self.model.train()
        data_set.train()
        max_steps = len(data_set)
        if self.max_steps_per_epoch is not None:
            max_steps = min(self.max_steps_per_epoch, max_steps)
        totals = {"Weight_loss": 0.0, "V_loss": 0.0, "P_loss": 0.0}
        count = 0
        frames = []
        for step, batch in enumerate(data_set):
            if step >= max_steps:
                break
            self.global_step += 1
            label = batch["label"].to(device).reshape(-1)
            pred = self._forward_batch(batch)
            if not torch.isfinite(pred).all():
                raise FloatingPointError("non-finite prediction; refusing to replace it with zero")
            # Keep the causal relation context intact; future label availability
            # only selects supervised residuals AFTER the complete forward pass.
            labelled = torch.isfinite(label)
            if not labelled.any():
                continue
            if self.use_weighted:
                descriptors = batch.get("predicates", {})
                if set(descriptors) != set(self.predicates):
                    raise ValueError("batch descriptors do not match the selected TSIL predicates")
                parts = self.criterion(pred[labelled], label[labelled],
                                       {name: values[labelled] for name, values in descriptors.items()})
                loss = parts["total"]
            else:
                loss = self.criterion(pred[labelled], label[labelled])
                parts = {"V_loss": loss, "P_loss": loss.new_zeros(())}
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            size = int(labelled.sum())
            count += size
            totals["Weight_loss"] += loss.item() * size
            totals["V_loss"] += parts["V_loss"].item() * size
            totals["P_loss"] += parts["P_loss"].item() * size
            frames.append(pd.DataFrame({"score": pred.detach().cpu().numpy(), "label": label.cpu().numpy()},
                          index=data_set.restore_index(batch["index"])))
        if not count:
            raise ValueError("training segment has no eligible samples")
        metrics = evaluate_daily(pd.concat(frames))
        metrics.update({key: value / count for key, value in totals.items()})
        metrics["Beta"] = float(np.mean(list(self.betas.values()))) if self.betas else 0.0
        return metrics["Weight_loss"], metrics

    def test_epoch(self, data_set, return_pred=False):
        self.model.eval()
        data_set.eval()
        frames = []
        elapsed = 0.0
        with torch.no_grad():
            for step, batch in enumerate(data_set):
                if step == 0:
                    self._forward_batch(batch)  # unmeasured warm-up
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                pred = self._forward_batch(batch)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - start
                label = batch["label"].to(device).reshape(-1)
                if not torch.isfinite(pred).all():
                    raise FloatingPointError("non-finite prediction during evaluation")
                frames.append(pd.DataFrame({"score": pred.cpu().numpy(), "label": label.cpu().numpy()},
                              index=data_set.restore_index(batch["index"])))
        if not frames:
            raise ValueError("evaluation segment has no eligible samples")
        preds = pd.concat(frames)
        preds = preds.reorder_levels(["datetime", "instrument"]).sort_index()
        metrics = evaluate_daily(preds)
        metrics["InfT_ms_per_sample"] = 1000.0 * elapsed / len(preds)
        metrics["InfT"] = metrics["InfT_ms_per_sample"]
        return metrics, preds if return_pred else None
    
    def fit(self, dataset, evals_result=None, evaluate_test=True):
        
        if evals_result is None:
            evals_result = {}
        names = ["train", "valid", "test"] if evaluate_test else ["train", "valid"]
        prepared = dataset.prepare(names)
        train_set, valid_set = prepared[:2]
        test_set = prepared[2] if evaluate_test else None

        best_score = float("-inf")
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
            if self.avg_params:
                self.model.load_state_dict(average_params(params_list["model"]))

            valid_metrics = self.test_epoch(valid_set)[0]
            evals_result["valid"].append(valid_metrics)
            self.logger.info("\tvalid metrics: %s" % valid_metrics)

            if self.eval_test and evaluate_test:
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
                    'Beta': train_metrics['Beta']
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

        if not math.isfinite(best_score):
            raise ValueError("validation IC is undefined for all epochs; no checkpoint can be selected")
        metrics, preds = self.test_epoch(test_set, return_pred=True) if evaluate_test else ({}, None)
        metrics['best_epoch'] = best_epoch
        metrics['best_score'] = best_score
        self.logger.info("test metrics: %s" % metrics)

        if self.logdir:
            self.logger.info("save model & pred to local directory")

            torch.save(best_params, self.logdir + "/model.bin")

            if preds is not None:
                preds.to_pickle(self.logdir + "/pred.pkl")
                if self.plot_training:
                    from qlib.contrib.report.analysis_model.analysis_model_performance import model_performance_graph
                    for i, fig in enumerate(model_performance_graph(preds, show_notebook=False)):
                        fig.write_html(self.logdir + f"/performance_{i}.html")

            metrics = {k: float(v) if isinstance(v, np.floating) else v for k, v in metrics.items()}
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
        _, preds = self.test_epoch(test_set, return_pred=True)
        return preds


def evaluate_daily(pred):
    """Metrics on finite target pairs; retain unlabeled scores for execution."""
    if not isinstance(pred.index, pd.MultiIndex) or "datetime" not in pred.index.names:
        raise ValueError("daily metrics require a datetime/instrument MultiIndex")
    prediction_count = len(pred)
    pred = pred[np.isfinite(pred.score) & np.isfinite(pred.label)]
    error = pred.score - pred.label
    daily = []
    for _, group in pred.groupby(level="datetime", sort=True):
        if len(group) < 2 or group.score.nunique() < 2 or group.label.nunique() < 2:
            daily.append((float("nan"), float("nan")))
        else:
            daily.append((group.score.corr(group.label), group.score.corr(group.label, method="spearman")))
    pairs = pd.DataFrame(daily, columns=["IC", "Rank_IC"])
    result = {"MSE": error.pow(2).mean(), "MAE": error.abs().mean(),
              "IC": pairs.IC.mean(), "Rank_IC": pairs.Rank_IC.mean(), "sample_count": len(pred),
              "prediction_count": prediction_count}
    for source, key in [("IC", "ICIR"), ("Rank_IC", "RankICIR")]:
        values = pairs[source].dropna()
        std = values.std(ddof=0)
        result[key] = values.mean() / std if std > 0 else float("nan")
    result["RankIC"] = result["Rank_IC"]
    return result

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
