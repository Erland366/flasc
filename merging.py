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

class FedProx(BaseMerging):
    def __init__(self, server_model: nn.Module, mu: float = 0.01):
        super().__init__(server_model)
        self.mu = mu
        
    def aggregate_updates(self, client_deltas: List[Dict[str, torch.Tensor]], **kwargs) -> Dict[str, torch.Tensor]:
        fed_avg = FedAvg(self.server_model)
        aggregate = fed_avg.aggregate_updates(client_deltas)
        
        for n, agg_delta in aggregate.items():
            agg_delta.mul_(1.0 / (1.0 + self.mu))
            
        return aggregate

class FedDare(BaseMerging):
    def __init__(self, server_model: nn.Module, p_drop: float = 0.5):
        super().__init__(server_model)
        self.p_drop = p_drop

    def aggregate_updates(self, client_deltas: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # We dropped several params and rescale the rest just like in Dropout
        batch_size = len(client_deltas)

        aggregate = {}

        for n, p in client_deltas[0].items():
            aggregate[n] = p.clone()

        # randomly drop some params based on bernoulli distribution
        for i in range(1, batch_size):
            for n, p in client_deltas[i].items():
                if torch.rand(1) < self.p:
                    aggregate[n] += p / (1 - self.p)

        # Then average
        for n in aggregate:
            aggregate[n] /= batch_size

        return aggregate
                


class MergingFactory:
    STRATEGY_PARAMS = {
        "fedavg": [],
        "fedprox" : ["mu"],
        "feddare" : ["p_drop"]
    }
    @staticmethod
    def get_merging_strategy(strategy: str, server_model: nn.Module, args=None, **kwargs) -> BaseMerging:
        strategy = strategy.lower()

        merging_kwargs = {}

        if args is not None:
            args_dict = vars(args) if hasattr(args, "__dict__") else args
            
            if strategy in MergingFactory.STRATEGY_PARAMS:
                for param in MergingFactory.STRATEGY_PARAMS[strategy]:
                    hyphen_param = param.replace("_", "-")
                    underscore_param = param.replace("-", "_")
                    
                    if hyphen_param in args_dict:
                        merging_kwargs[param] = args_dict[hyphen_param]
                    elif underscore_param in args_dict:
                        merging_kwargs[param] = args_dict[underscore_param]
            
            prefix_pattern = f"{strategy}_"
            hyphen_prefix_pattern = f"{strategy}-"
            
            for key, value in args_dict.items():
                if key.startswith(prefix_pattern):
                    param_name = key[len(prefix_pattern):]
                    merging_kwargs[param_name] = value
                elif key.startswith(hyphen_prefix_pattern):
                    param_name = key[len(hyphen_prefix_pattern):]
                    merging_kwargs[param_name] = value
        
        merging_kwargs.update(kwargs)
        
        if strategy == "fedavg":
            return FedAvg(server_model)
        elif strategy == "fedprox":
            # TODO: Double check when we have the args + kwargs (pass the params through argparse)
            mu = merging_kwargs.get("mu", 0.01)
            return FedProx(server_model, mu=mu)
        elif strategy == "feddare":
            p_drop = merging_kwargs.get("p_drop", 0.5)
            return FedDare(server_model, p_drop=p_drop)
        else:
            raise ValueError(f"Unknown merging strategy: {strategy}")