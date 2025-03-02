import torch

from torch import nn

class BaseMerging:
    def __init__(self, server_model: nn.Module):
        self.server_model = server_model
        self.server_params = {n: p for n, p in server_model.named_parameters() if p.requires_grad}

    def aggregate_updates(self, client_deltas: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    def update_server_model(self, aggregated_update: dict[str, torch.Tensor], server_opt: torch.optim.Optimizer) -> None:
        server_opt.zero_grad()
        for n, p in self.server_params.items():
            if n in aggregated_update:
                p.grad = aggregated_update[n]
        server_opt.step()

class FedAvg(BaseMerging):
    def aggregate_updates(self, client_deltas: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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
        
    def aggregate_updates(self, client_deltas: list[dict[str, torch.Tensor]], **kwargs) -> dict[str, torch.Tensor]:
        fed_avg = FedAvg(self.server_model)
        aggregate = fed_avg.aggregate_updates(client_deltas)
        
        # fedprox was applied to the loss function, removed from here
        # for n, agg_delta in aggregate.items():
        #     agg_delta.mul_(1.0 / (1.0 + self.mu))
            
        return aggregate

class FedDare(BaseMerging):
    def __init__(self, server_model: nn.Module, p_drop: float = 0.5):
        super().__init__(server_model)
        self.p_drop = p_drop

    def aggregate_updates(self, client_deltas: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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
        
        if strategy in ["fedavg", "fedprox", "fedavgm", "fedadam", "fedadagrad", "fedyogi", "scaffold"]:
            return FedAvg(server_model)
        # elif strategy == "fedprox":
        #     # TODO: Double check when we have the args + kwargs (pass the params through argparse)
        #     mu = merging_kwargs.get("mu", 0.01)
        #     return FedProx(server_model, mu=mu)
        elif strategy == "feddare":
            p_drop = merging_kwargs.get("p_drop", 0.5)
            return FedDare(server_model, p_drop=p_drop)
        else:
            raise ValueError(f"Unknown merging strategy: {strategy}")


# from FedLLM Bench: https://github.com/rui-ye/FedLLM-Bench/blob/main/main_sft.py
def global_aggregate(fed_args, global_dict, delta_dict_list, sample_num_list, clients_this_round, round_idx, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None, lr=1):
    """
    Args:
        fed_args: the "args" in our experiments
    """
    sample_this_round = sum(sample_num_list)
    clients_this_round = range(clients_this_round)
    global_auxiliary = None
    
    server_params = fed_args.server_hparams
    if server_params is not None:
        beta1 = server_params["beta1"] if "beta1" in server_params else None
        beta2 = server_params["beta2"] if "beta2" in server_params else None
        tau = server_params["tau"] if "tau" in server_params else None

    if fed_args.merging_strategy == 'scaffold':
        for key in global_dict.keys():
            global_dict[key].data -= lr * sum([delta_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_deltas = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_deltas[client][key] for client in clients_this_round]) 
            global_auxiliary[key] += delta_auxiliary / len(clients_this_round)
    
    elif fed_args.merging_strategy == 'fedavgm':
        # Momentum-based FedAvg
        for key in global_dict.keys():
            delta_w = sum([delta_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
            proxy_dict[key] = beta1 * proxy_dict[key] + (1 - beta1) * delta_w if round_idx > 0 else delta_w
            global_dict[key].data -= proxy_dict[key]

    elif fed_args.merging_strategy == 'fedadagrad':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([delta_dict_list[client][key] for client in clients_this_round]) / len(clients_this_round)
            # In paper 'adaptive federated optimization', momentum is not used
            proxy_dict[key] = delta_w  
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            global_dict[key].data -= lr * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+tau)

    elif fed_args.merging_strategy == 'fedyogi':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([delta_dict_list[client][key] for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = beta1 * proxy_dict[key] + (1 - beta1) * delta_w if round_idx > 0 else delta_w
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (1-beta2)*delta_square*torch.sign(param - delta_square)
            global_dict[key].data -= lr * torch.div(proxy_dict[key]/(1-beta1**(round_idx+1)), torch.sqrt(opt_proxy_dict[key]/(1-beta2**(round_idx+1)))+tau)

    elif fed_args.merging_strategy == 'fedadam':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([delta_dict_list[client][key] for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = beta1 * proxy_dict[key] + (1 - beta1) * delta_w  # if round_idx > 0 else delta_w
            opt_proxy_dict[key] = beta2* opt_proxy_dict[key] + (1-beta2)*torch.square(delta_w)  # if round_idx > 0 else torch.square(delta_w)

            global_dict[key].data -= lr * torch.div(proxy_dict[key]/(1-beta1**(round_idx+1)), torch.sqrt(opt_proxy_dict[key]/(1-beta2**(round_idx+1)))+tau)

    else:   # Normal dataset-size-based aggregation 
        for key in global_dict.keys():
            global_dict[key].data -= lr * sum([delta_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    
    return global_dict, global_auxiliary