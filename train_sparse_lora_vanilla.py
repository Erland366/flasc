from __future__ import annotations

import os
import time
import argparse
from tqdm import tqdm
from copy import deepcopy
from typing import Union, List, Iterable, Callable, Literal

from dotenv import load_dotenv

load_dotenv()

def str2bool(s):
    return s.lower() == 'true'

def parse():
    parser   = argparse.ArgumentParser()
    parser.add_argument('--gpu',            default='0',        type=str)
    parser.add_argument('--dir',            default='runs',     type=str)
    parser.add_argument('--name',           default='test',     type=str)
    parser.add_argument('--save',           default='false',     type=str)
    parser.add_argument('--dataset',        default='cifar10',  type=str)
    parser.add_argument('--iid-alpha',      default=0.1,  type=float)
    parser.add_argument('--clients',        default=500,       type=int)
    parser.add_argument('--model',          default='vit_b_16', type=str)
    parser.add_argument('--resume',         default=0,          type=int)
    parser.add_argument('--seed',           default=0,          type=int)
    parser.add_argument('--eval-freq',      default=10,         type=int)
    parser.add_argument('--eval-first',  default='false',      type=str)
    parser.add_argument('--eval-frac',  default=1,        type=float)
    parser.add_argument('--eval-masked',  default='true',      type=str)
    #
    parser.add_argument('--server-opt',       default='adam',  type=str)
    parser.add_argument('--server-lr',        default=1e-3,    type=float)
    parser.add_argument('--server-batch',     default=10,       type=int)
    parser.add_argument('--server-rounds',    default=200,      type=int)
    parser.add_argument('--client-lr',        default=1e-3,    type=float)
    parser.add_argument('--client-batch',     default=16,      type=int)
    parser.add_argument('--client-epochs',    default=1,       type=int)
    parser.add_argument('--client-freeze',    default='false',     type=str)
    parser.add_argument('--server-freeze',    default='false',     type=str)
    parser.add_argument('--freeze-a',         default='false',     type=str)
    parser.add_argument('--lora-rank',  default=16, type=int)
    parser.add_argument('--lora-alpha', default=16, type=int)
    parser.add_argument('--l2-clip-norm', default=0, type=float)
    parser.add_argument('--noise-multiplier', default=0, type=float)
    parser.add_argument('--use_wandb', action="store_true", default=True)
    parser.add_argument('--use_tensorboard', action="store_true", default=True)
    parser.add_argument("--project_name", default="flasc", type=str)
    parser.add_argument("--entity", default=None, type=str)

    parser.add_argument("--merging_strategy", default="fedavg", type=str)
    return parser.parse_args()

def main():
    args = parse()
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['ROCR_VISIBLE_DEVICES'] = args.gpu
    # os.environ['HIP_VISIBLE_DEVICES'] = args.gpu
    if args.gpu != '0':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import random
    import numpy as np
    import torch
    import tensorflow as tf
    from torch import nn
    tf.config.set_visible_devices([], device_type='GPU')
    print(f"Visible GPUs: {torch.cuda.device_count()}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def get_topk_mask(x, density):
        mask = torch.zeros_like(x).bool()
        k = int(x.numel()*density)
        _, keep_idx = torch.topk(x, k=k)
        mask[keep_idx] = 1
        return mask

    from train_utils import log_stats, login_wandb, is_wandb_available
    from merging import MergingFactory


    def fl_train(
        save: bool, 
        run_dir: Union[str, os.PathLike],
        server_model: nn.Module, 
        clients: List[int], 
        valloader: Iterable, 
        testloader: Iterable, 
        test_batch: Callable,
        rounds: int, 
        eval_freq: int, 
        eval_first: bool, 
        eval_masked: bool, 
        server_opt: Literal["sgd", "adam"],
        server_batch: int, 
        server_lr: float, 
        server_freeze: bool, 
        client_lr: float,
        client_epochs: int, 
        client_freeze: bool, 
        l2_clip_norm: float=0.0,
        merging_strategy: str="fedavg",
    ):
        if args.use_tensorboard:
            writer = tf.summary.create_file_writer(run_dir)
        else:
            writer = None

        if args.use_wandb:
            if not is_wandb_available():
                raise ValueError("Wandb is not available")
            
            import wandb

            login_wandb()
            wandb.init(
                project=args.project_name,
                entity=args.entity,
                name=args.wandb_name,
                config=vars(args),
            )

        pbar = tqdm(range(rounds))

        server_params = {n:p for n,p in server_model.named_parameters() if p.requires_grad}
        server_mask = {n:torch.ones_like(p) for n,p in server_params.items()}

        if server_freeze:
            for p in server_params.values():
                p.requires_grad = False

        if server_opt == 'sgd':
            server_opt = torch.optim.SGD(server_params.values(), lr=server_lr)
        elif server_opt == 'adam':
            server_opt = torch.optim.AdamW(server_params.values(), lr=server_lr)
        else:
            raise ValueError()
        sched = torch.optim.lr_scheduler.StepLR(server_opt, step_size=1, gamma=1)

        merger = MergingFactory.get_merging_strategy(merging_strategy, server_model)

        eval_accu = 0
        def eval_loop(model, loader):        
            model.eval()
            stats_acc = {}
            for x,y in loader:
                with torch.no_grad():
                    _, stats = test_batch(model, x, y)
                for k,v in stats.items():
                    stats_acc[k] = stats_acc.get(k, 0) + v
            stats_acc['loss'] /= stats_acc['count']
            return stats_acc
        
        if eval_first:
            # I think this is still error?
            log_stats(writer, "eval", stats, 0)
        
        for rnd in pbar:
            client_deltas = []
            stats_acc = {}
            client_ids = torch.randperm(len(clients))[:server_batch]

            for i,client_id in enumerate(client_ids):
                # Download Sparsity
                client_model = deepcopy(server_model)

                # Local Training
                client_opt = torch.optim.SGD(client_model.parameters(), lr=client_lr, momentum=0.9)
                client_loader = clients[client_id]
                client_acc = {}
                for epoch in range(client_epochs):
                    for x,y in client_loader:
                        loss, stats = test_batch(client_model, x, y)
                        client_opt.zero_grad()
                        loss.backward()

                        if l2_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(client_model.parameters(), l2_clip_norm)

                        client_opt.step()
                        for k,v in stats.items():
                            client_acc[k] = client_acc.get(k, 0) + v
                        pbar.set_description(f"eval: {eval_accu} | client {i}, epoch {epoch} | loss {loss:.4f}")

                # This is our delta parameter
                neg_client_delta = {
                    n: server_params[n].data - cp.data for n,cp 
                                    in client_model.named_parameters() if cp.requires_grad
                }

                client_deltas.append(neg_client_delta)

                # Log last iteration
                client_acc['norm'] = 0
                for k,v in client_acc.items():
                    stats_acc[k] = stats_acc.get(k, 0) + v

            # Optimizer step
            aggregated_update = merger.aggregate_updates(client_deltas)
            merger.update_server_model(aggregated_update, server_opt)
            sched.step()

            # Eval and Logging
            if (rnd+1) % eval_freq == 0:
                eval_model = deepcopy(server_model)
                if valloader is not None:
                    log_stats(writer, "eval", eval_loop(eval_model, valloader), rnd+1)
                log_stats(writer, "test", eval_loop(eval_model, testloader), rnd+1)
            
            stats_acc['norm'] /= server_batch
            stats_acc['loss'] /= stats_acc['count']
            log_stats(writer, "train", stats_acc, rnd+1)

            pbar.set_description(f"eval: {eval_accu}")
        # if save:
        #     torch.save({'delta': server_params, 'mask': server_mask}, f"{run_dir}/save.pt")

    import data_utils
    clients, valloader, testloader, test_batch = data_utils.build_dataset(
        args.dataset, args.client_batch, args.clients, args.iid_alpha, args.seed, args.eval_frac)

    import models
    model = models.build_model(args.dataset)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    models.add_adapters_dataset(args.dataset, model, args.lora_rank, args.lora_alpha)
    if str2bool(args.freeze_a):
        for n,p in model.named_parameters():
            if "lora_A" in n:
                p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {trainable} parameters ({100*trainable/total:.2f}% of original {total})")
    model = model.cuda()

    # Add hash4 at the end?
    args.name += f"_{args.model}_c{args.clients}_b{args.client_batch}_lr{args.client_lr}_e{args.client_epochs}_{args.merging_strategy}"
    args.wandb_name = f"{args.name}"
    # add name with date
    args.run_dir_name = f"{args.dir}/{args.name}_{time.strftime('%Y%m%d-%H%M%S')}"

    run_dir = args.run_dir_name
    os.makedirs(run_dir)
    import json

    with open(f"{run_dir}/args.json", 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved args to {run_dir}")

    fl_train(str2bool(args.save),
        run_dir, model, clients, valloader, testloader, test_batch,
        rounds=args.server_rounds, 
        eval_freq=args.eval_freq,
        eval_first=str2bool(args.eval_first),
        eval_masked=str2bool(args.eval_masked),
        server_opt=args.server_opt,
        server_batch=args.server_batch, 
        server_lr=args.server_lr,
        server_freeze=str2bool(args.server_freeze),
        client_lr=args.client_lr, 
        client_epochs=args.client_epochs,
        client_freeze=str2bool(args.client_freeze),
        l2_clip_norm=args.l2_clip_norm,
        merging_strategy=args.merging_strategy
    )

    if is_wandb_available():
        import wandb

        if wandb.run is not None:
            wandb.finish()  

if __name__ == "__main__":
    main()
