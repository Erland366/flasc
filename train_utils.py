from __future__ import annotations

import os

import torch
import tensorflow as tf

from typing import Union
from loguru import logger

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