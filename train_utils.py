from __future__ import annotations

import os

import torch
import tensorflow as tf

from typing import Union
from loguru import logger
from copy import deepcopy

sig = torch.nn.Sigmoid()

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
    if fed_args.merging_strategy in ['fedadagrad', 'fedyogi', 'fedadam']:
        proxy_dict, opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
            opt_proxy_dict[key] = torch.zeros_like(global_dict[key])  * fed_args.server_hparams["tau"]**2
    elif fed_args.merging_strategy == 'fedavgm':
        proxy_dict = {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
    return proxy_dict, opt_proxy_dict

def get_auxiliary_dict(fed_args, global_dict):

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