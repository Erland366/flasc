from __future__ import annotations

import os

import torch
from torch.optim.optimizer import Optimizer
import tensorflow as tf

from typing import Union
from loguru import logger
from copy import deepcopy

sig = torch.nn.Sigmoid()


class Yogi(Optimizer):
    r"""Implements Yogi algorithm.
     proposed in "Adaptive Methods for Nonconvex Optimization" (ICML 2018).

     ps. I don't konw why the one in torch_optimizer performs so poorly,
     so I implement a new one below. It works much better.
     I guess the difference is from tiny issues in the scaling factor used
     at early training.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (float, optional): coefficient used for computing running averages of gradient (default: 0.9)
        beta2 (float, optional): coefficient used for computing running averages of squared gradient (default: 0.999)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super(Yogi, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError("Yogi does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Update biased first moment estimate.
                state["exp_avg"] = beta1 * state["exp_avg"] + (1 - beta1) * grad if t > 1 else grad
                # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update the second moment estimate using Yogi update:
                # v_t = v_{t-1} - (1 - beta2) * sign(v_{t-1} - grad^2) * grad^2
                grad_square = torch.square(state["exp_avg"])
                state["exp_avg_sq"] = state["exp_avg_sq"] - (1-beta2)*grad_square*torch.sign(state["exp_avg_sq"] - grad_square)

                p.data -= lr * torch.div(state["exp_avg"]/(1-beta1**t),
                                         torch.sqrt(state["exp_avg_sq"]/(1-beta2**t))+eps)

        return loss


def is_wandb_available():
    try:
        import wandb
        return True
    except ImportError:
        return False

def login_wandb(environ_name: str = "WANDB_API_KEY", token: str | None = None, **kwargs):
    if not is_wandb_available():
        raise ImportError("wandb is not available. Please install it via `pip install wandb`.")
    import wandb

    if token is None:
        token = os.getenv(environ_name)
        logger.debug(f"Use token from environment variable {environ_name}")
    wandb.login(key=token, **kwargs)

def test_batch_cls(model, x, y, multilabel=False): # classification
    outputs = model(x, labels=y)
    logits = outputs.logits.detach()
    loss = outputs.loss # huggingface loss is already averaged
    if multilabel: # label set is a binary vector
        preds = torch.where(sig(logits) < 0.5, 0, 1)
        stats = {
            'tp': (preds*y).sum().item(),
            'tn': ((1-preds)*(1-y)).sum().item(),
            'fp': (preds*(1-y)).sum().item(),
            'fn': ((1-preds)*y).sum().item(),
            'count': x.shape[0],
            'loss': loss.item()*x.shape[0],
        }
    else: # labels are integers
        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()
        stats = {
            'tp': correct,
            'fp': len(y) - correct,
            'fn': len(y) - correct,
            'count': x.shape[0],
            'loss': loss.item()*x.shape[0],
        }
    return loss, stats

def test_batch_nwp(model, x): # next word (token) prediction
    non_pad_idx = x[:, 1:] != 50256                       # [B, S]: bool
    total = non_pad_idx.sum().item()                      # [sentences]: int
    output = model(x)
    logits = output.logits[:, :-1]
    flat_logits = logits.reshape(-1, 50257) # exclude last token
    loss = torch.nn.functional.nll_loss(
        torch.nn.functional.log_softmax(flat_logits, dim=-1), # flat predictions
        x[:, 1:].reshape(-1), # flat tokens
        ignore_index=50256,
        reduction='sum') / total
    with torch.no_grad():
        pred_toks = logits.argmax(dim=-1)                 # [sentences, tokens]: 0...50256
        correct_toks = pred_toks == x[:, 1:]              # [sentences, tokens]: bool
        correct = (non_pad_idx*correct_toks).sum().item() # [sentences]: int
        stats = {
            'tp': correct, 
            'fp': total - correct, 
            'fn': total - correct,
            'count': total,
            'loss': loss.item()*total,
        }
    return loss, stats

def get_metric(stats, metric):
        if stats['tp'] == 0:
            return 0
        elif metric == 'accu':
            return stats['tp'] / (stats['tp'] + stats['fp'])
        elif metric == 'recall':
            return stats['tp'] / (stats['tp'] + stats['fn'])
        elif metric == 'f1':
            return 2*stats['tp'] / (2*stats['tp'] + stats['fp'] + stats['fn'])

def log_stats(writer: Union[None, "SummaryWriter"], prefix, stats, step):

    if writer is not None:
        with writer.as_default():
            tf.summary.scalar(f"{prefix}/accuracy", get_metric(stats, 'accu'), step=step)
            tf.summary.scalar(f"{prefix}/recall", get_metric(stats, 'recall'), step=step)
            tf.summary.scalar(f"{prefix}/f1", get_metric(stats, 'f1'), step=step)
            for k,v in stats.items():
                if k not in ['tp', 'fp', 'tn', 'fn']:
                    tf.summary.scalar(f"{prefix}/{k}", v, step=step)

    # Wandb logs
    if not is_wandb_available():
        return

    import wandb
    wandb.log({f"{prefix}/accuracy": get_metric(stats, 'accu')}, step=step)
    wandb.log({f"{prefix}/recall": get_metric(stats, 'recall')}, step=step)
    wandb.log({f"{prefix}/f1": get_metric(stats, 'f1')}, step=step)
    for k,v in stats.items():
        if k not in ['tp', 'fp', 'tn', 'fn']:
            wandb.log({f"{prefix}/{k}" : v}, step=step)

def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters
    """
    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())
    
    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) 
        for i in range(len(tensor_1))])
    
    return norm

# from: https://github.com/rui-ye/FedLLM-Bench/blob/main/federated_learning/fed_utils.py
def get_proxy_dict(fed_args, global_dict):
    opt_proxy_dict = None
    proxy_dict = None
    if fed_args.merging_strategy in ['fedadagrad', 'fedyogi', 'fedadam', 'fedprox']:
        proxy_dict, opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
            opt_proxy_dict[key] = torch.zeros_like(global_dict[key])  * fed_args.server_hparams["tau"]**2  # why we have tau here?
    elif fed_args.merging_strategy == 'fedavgm':
        proxy_dict = {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
    return proxy_dict, opt_proxy_dict

def get_auxiliary_dict(fed_args, global_dict):
    """Get the auxiliary model and the auxiliary deltas for SCAFFOLD
    """

    if fed_args.merging_strategy in ['scaffold']:
        global_auxiliary = {}               # c in SCAFFOLD
        for key in global_dict.keys():
            global_auxiliary[key] = torch.zeros_like(global_dict[key])
        auxiliary_model_list = [deepcopy(global_auxiliary) for _ in range(fed_args.clients)]    # c_i in SCAFFOLD
        auxiliary_deltas = []    # delta c_i in SCAFFOLD

    else:
        global_auxiliary = None
        auxiliary_model_list = [None]*fed_args.clients
        auxiliary_deltas = []

    return global_auxiliary, auxiliary_model_list, auxiliary_deltas
