from typing import List, Dict

import torch

from torch import nn

class BaseMerging:
    def __init__(self, server_model: nn.Module):
        self.server_model = server_model
        self.server_params = {n: p for n, p in server_model.named_parameters() if p.requires_grad}

    def aggregate_updates(self, client_deltas: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def update_server_model(self, aggregated_update: Dict[str, torch.Tensor], server_opt: torch.optim.Optimizer) -> None:
        server_opt.zero_grad()
        for n, p in self.server_params.items():
            if n in aggregated_update:
                p.grad = aggregated_update[n]
        server_opt.step()

class FedAvg(BaseMerging):
    def aggregate_updates(self, client_deltas: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(client_deltas)

        aggregate = {}
        for n, p in client_deltas[0].items():
            aggregate[n] = p.clone()

        for i in range(1, batch_size):
            for n, p in client_deltas[i].items():
                aggregate[n] += p

        for n in aggregate:
            aggregate[n] /= batch_size
        
        return aggregate

class MergingFactory:
    @staticmethod
    def get_merging_strategy(strategy: str, server_model: nn.Module, *args, **kwargs) -> BaseMerging:
        if strategy == 'fedavg':
            return FedAvg(server_model, *args, **kwargs)
        else:
            raise ValueError(f"Unknown merging strategy: {strategy}")