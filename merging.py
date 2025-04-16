from typing import List, Dict
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

import torch

from torch import nn


class BaseMerging:
    def __init__(self, server_model: nn.Module):
        self.server_model = server_model
        self.server_params = {n: p for n, p in server_model.named_parameters() if p.requires_grad}

    def aggregate_updates(self, neg_client_deltas: List[Dict[str, torch.Tensor]],
                          average_weights: torch.Tensor,  # shape: (num_clients,)
                          **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def update_server_model(self, aggregated_update: Dict[str, torch.Tensor], server_opt: torch.optim.Optimizer) -> None:
        server_opt.zero_grad()
        for n, p in self.server_params.items():
            if n in aggregated_update:
                p.grad = aggregated_update[n]
        server_opt.step()


class TaskArithmetic(BaseMerging):
    def aggregate_updates(self, neg_client_deltas: List[Dict[str, torch.Tensor]],
                          average_weights: torch.Tensor,  # shape: (num_clients,)
                          scaling_coefficient: float = 1.,  # set to 1.0 is FedAvg
                          **kwargs
                          ) -> Dict[str, torch.Tensor]:
        num_clients = len(neg_client_deltas)
        aggregate = {}
        for n, p in neg_client_deltas[0].items():
            aggregate[n] = p.clone() * scaling_coefficient * average_weights[0]

        for i in range(1, num_clients):
            for n, p in neg_client_deltas[i].items():
                aggregate[n] += p * scaling_coefficient * average_weights[i]
        return aggregate


class FisherMerging(BaseMerging):
    def get_param_squared_gradients(self, model: nn.Module):
        """
        get the squared gradients of parameters
        :param model: nn.Module, model
        :return:
        """
        param_squared_gradients = {param_name: param_value.grad.detach().cpu().data ** 2
                                   for param_name, param_value in model.named_parameters()
                                   if param_value.requires_grad}
        return param_squared_gradients

    def get_models_fisher_norm(self, models_to_merge_param_dict: dict,
                               models_to_merge_fisher_weights_list: list):
        """
        get normalization of fisher weights of all the models that need to be merged
        :param models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
        value is a list of the corresponding parameters of all the models that need to be merged
        :param models_to_merge_fisher_weights_list: list, list of dictionaries with length len(models_to_merge),
        each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
        :return:
        """
        # dict, key is parameter name, value is a Tensor with shape (num_models_to_merge, )
        models_fisher_norm_dict = {}
        # compute L2 norm over models for each parameter
        for param_name, _ in models_to_merge_param_dict.items():
            # Tensor, shape (num_models_to_merge, *fisher_weight_shape)
            models_fisher = torch.stack([model_to_merge_fisher_weights[param_name] for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list], dim=0)
            dims = [dim_idx for dim_idx in range(1, models_fisher.dim())]
            # Tensor, shape (num_models_to_merge, ), compute L2 norm for each parameter
            models_fisher_norm = torch.norm(models_fisher, dim=dims)
            models_fisher_norm_dict[param_name] = models_fisher_norm

        # Tensor, shape (num_models_to_merge, num_parameters)
        models_fisher_norm = torch.stack([models_fisher_norm for models_fisher_norm in models_fisher_norm_dict.values()], dim=1)
        # Tensor, shape (num_models_to_merge, ), compute L2 norm over all the parameters
        models_fisher_norm = torch.norm(models_fisher_norm, dim=1)
        return models_fisher_norm

    def merging_with_fisher_weights(self, models_to_merge_param_dict: dict,
                                    models_to_merge_fisher_weights_list: list,
                                    fisher_scaling_coefficients: torch.Tensor,
                                    normalize_fisher_weight: bool = True, minimal_fisher_weight: float = 1e-6):
        """
        merge parameters of different models with computed fisher weights
        :param models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
        value is a list of the corresponding parameters of all the models that need to be merged
        :param models_to_merge_fisher_weights_list: list, list of dictionaries with length len(models_to_merge),
        each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
        :param fisher_scaling_coefficients: torch.Tensor, scaling coefficients to merge fisher weights
        :param normalize_fisher_weight: boolean, whether to normalize fisher weights (L2 norm) or not
        :param minimal_fisher_weight: float, the minimal value in fisher weights, used for tackling the potential numerical issues
        :return:
        """
        # move to cpu to save memory
        for k, v in models_to_merge_param_dict.items():
            for i in range(len(v)):
                v[i] = v[i].cpu()
        for models_to_merge_fisher_weights in models_to_merge_fisher_weights_list:
            for k, v in models_to_merge_fisher_weights.items():
                models_to_merge_fisher_weights[k] = v.cpu()

        # dict, dictionary of model parameters
        merged_params = {}

        if normalize_fisher_weight:
            # Tensor, shape (num_models_to_merge, ), L2 norm over all the parameters of models that need to be merged
            models_fisher_norm = self.get_models_fisher_norm(models_to_merge_param_dict=models_to_merge_param_dict,
                                                        models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list)

        for param_name, param_value_list in models_to_merge_param_dict.items():
            # shape (num_models_to_merge, *parameter_shape)
            # for i in range(len(param_value_list)):
            #     param_value_list[i] = param_value_list[i].detach().data
            param_values = torch.stack(param_value_list, dim=0)
            # Tensor, shape (num_models_to_merge, *fisher_weight_shape), use minimal_fisher_weight to solve the potential numerical issues
            models_to_merge_fisher_weights = torch.stack([model_to_merge_fisher_weights[param_name]
                                                            for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list], dim=0) + minimal_fisher_weight

            # Tensor, shape (num_models_to_merge, 1, 1, ...)
            reshaped_scaling_coefficients = fisher_scaling_coefficients.reshape(-1, *[1 for _ in range(param_values.dim() - 1)]).to(param_values.device)

            if normalize_fisher_weight:
                # Tensor, shape (num_models_to_merge, )
                _models_fisher_norm = 1.0 / (models_fisher_norm + minimal_fisher_weight)
                normalized_models_fisher_norm = _models_fisher_norm / _models_fisher_norm.sum()
                normalized_models_fisher_norm = normalized_models_fisher_norm.reshape(-1, *[1 for _ in range(param_values.dim() - 1)])
                reshaped_scaling_coefficients = reshaped_scaling_coefficients * normalized_models_fisher_norm

            # shape (*parameter_shape)
            numerator = (reshaped_scaling_coefficients * models_to_merge_fisher_weights * param_values).sum(dim=0)

            # shape (*parameter_shape)
            denominator = (reshaped_scaling_coefficients * models_to_merge_fisher_weights).sum(dim=0)

            merged_param = numerator / denominator
            merged_params[param_name] = nn.Parameter(merged_param)
        return merged_params

    def aggregate_updates(self, neg_client_deltas: List[Dict[str, torch.Tensor]],
                          average_weights: torch.Tensor,  # shape: (num_clients,)
                          client_loaders: List[torch.utils.data.DataLoader],  # shape: (num_clients,)
                          nums_fisher_examples: torch.Tensor,  # shape: (num_clients,)
                          device: torch.device,
                          normalize_fisher_weight: bool = True,
                          minimal_fisher_weight: float = 1e-6,
                          **kwargs
                          ) -> Dict[str, torch.Tensor]:
        models_to_merge_param_dict = defaultdict(list)
        models_to_merge_fisher_weights_list = []
        for client_idx, (client_loader, num_fisher_examples) in enumerate(zip(client_loaders,
                                                                              nums_fisher_examples)):
            # obtain the client model
            model_to_merge = deepcopy(self.server_model)
            neg_client_delta = neg_client_deltas[client_idx]
            for n, p in model_to_merge.named_parameters():
                if n in neg_client_delta:
                    p.data -= neg_client_delta[n]
            param_dict = {param_name: param_value for param_name, param_value in model_to_merge.named_parameters()}
            param_names_to_merge = [param_name for param_name, param_value in model_to_merge.named_parameters() if param_value.requires_grad]
            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(param_dict[param_name])
            # get the fisher weights of the client model trainable parameters
            model_to_merge.to(device)
            batches_fisher_weights_list = []

            num_computed_examples = 0
            train_dataloader = client_loader
            batch_size = train_dataloader.batch_size
            if num_fisher_examples % batch_size != 0:
                print(f"warning: the number of examples for computing fisher cannot be fully divided by the batch size for client {client_idx}, "
                      "which may lead to a slightly different number of the actually used examples.")
            for step, inputs in tqdm(enumerate(train_dataloader), desc=f"computing fisher weights for client {client_idx}"):
                if num_computed_examples >= num_fisher_examples:
                    break
                x, y = inputs[0].to(device), inputs[1].to(device)
                outputs = model_to_merge(x, labels=y)  # TODO: this is only for classification
                # Tensor, shape (batch_size, num_label_classes)
                logits = outputs.logits
                # compute fisher weights for regression task
                if logits.shape[-1] == 1:
                    # use the label information to compute loss and obtain gradients
                    mse_loss = outputs.loss
                    model_to_merge.zero_grad()
                    mse_loss.backward()
                    # dict, fisher weights of a batch
                    batch_fisher_weights = self.get_param_squared_gradients(model=model_to_merge)
                # compute fisher weights for classification task
                else:
                    # use detach() to detach from the computation graph
                    # Tensor, shape (batch_size, num_label_classes)
                    labels_probabilities = torch.softmax(logits, dim=-1).detach()
                    labels_log_probabilities = torch.log_softmax(logits, dim=-1)
                    # sqrt labels_probabilities, since torch.sqrt(labels_probabilities) would be squared in the following squared gradients
                    labels_expectations = torch.sqrt(labels_probabilities) * labels_log_probabilities
                    # sum over label classes and batch dimension
                    sum_labels_expectations = labels_expectations.sum(dim=-1).sum(dim=0)
                    model_to_merge.zero_grad()
                    sum_labels_expectations.backward()
                    # dict, fisher weights of a batch
                    batch_fisher_weights = self.get_param_squared_gradients(model=model_to_merge)

                batches_fisher_weights_list.append(batch_fisher_weights)
                num_computed_examples += batch_size

            model_to_merge_fisher_weights = {}
            for batch_fisher_weights in batches_fisher_weights_list:
                for key in batch_fisher_weights:
                    if key not in model_to_merge_fisher_weights:
                        model_to_merge_fisher_weights[key] = batch_fisher_weights[key]
                    else:
                        model_to_merge_fisher_weights[key] += batch_fisher_weights[key]

            # mean over batches
            for key in model_to_merge_fisher_weights:
                model_to_merge_fisher_weights[key] /= num_computed_examples
            models_to_merge_fisher_weights_list.append(model_to_merge_fisher_weights) 

        # merging with fisher weights
        # if fisher_scaling_coefficients is None, then set the fisher weights of different models to contribute equally
        fisher_scaling_coefficients = average_weights
        # merging with fisher weights
        merged_params = self.merging_with_fisher_weights(models_to_merge_param_dict=models_to_merge_param_dict,
                                                         models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list,
                                                         fisher_scaling_coefficients=fisher_scaling_coefficients,
                                                         normalize_fisher_weight=normalize_fisher_weight,
                                                         minimal_fisher_weight=minimal_fisher_weight)
        # obtain the aggregated negative delta
        aggregate = {}
        for n, p in merged_params.items():
            aggregate[n] = self.server_params[n] - p
        return aggregate


class TiesMerging(BaseMerging):
    def aggregate_updates(self,
                          neg_client_deltas: List[Dict[str, torch.Tensor]],
                          average_weights: torch.Tensor,  # shape: (num_clients,)
                          scaling_coefficient: float = 1.,
                          **kwargs
                          ) -> Dict[str, torch.Tensor]:
        # get the finetuned models
        # directly scale the deltas for weights averaging
        models_to_merge = []
        for i, neg_client_delta in enumerate(neg_client_deltas):
            client_model = deepcopy(self.server_model)
            for n, p in client_model.named_parameters():
                if n in neg_client_delta:
                    p.data -= neg_client_delta[n]
            models_to_merge.append(client_model)
        from model_merging_methods.merging_methods import MergingMethod
        ties_merging_method = MergingMethod("ties_merging")
        merged_params = ties_merging_method.ties_merging(self.server_model, models_to_merge,
                                         param_value_mask_rate=kwargs.get("param_value_mask_rate", 0.8),
                                         scaling_coefficient=scaling_coefficient)
        # obtain the aggregated negative delta
        aggregate = {}
        for n, p in merged_params.items():
            aggregate[n] = self.server_params[n] - p
        return aggregate


class RegmeanMerging(BaseMerging):
    def aggregate_updates(self,
                          neg_client_deltas: List[Dict[str, torch.Tensor]],
                          average_weights: torch.Tensor,  # shape: (num_clients,)
                          nums_regmean_examples: torch.Tensor,
                          client_loaders: List[torch.utils.data.DataLoader],
                          device: torch.device,
                          **kwargs
                          ) -> Dict[str, torch.Tensor]:
        # get the finetuned models
        models_to_merge = []
        for neg_client_delta in neg_client_deltas:
            client_model = deepcopy(self.server_model)
            for n, p in client_model.named_parameters():
                if n in neg_client_delta:
                    p.data -= neg_client_delta[n]
            models_to_merge.append(client_model)
        from model_merging_methods.merging_methods import MergingMethod
        regmean_merging_method = MergingMethod("regmean_merging")
        merged_params = regmean_merging_method.regmean_merging(models_to_merge,
                                                               average_weights=average_weights,
                                                               train_loaders=client_loaders,
                                                               exclude_param_names_regex=[],
                                                               nums_regmean_examples=nums_regmean_examples,
                                                               device=device)
        # obtain the aggregated negative delta
        aggregate = {}
        for n, p in merged_params.items():
            aggregate[n] = self.server_params[n] - p
        return aggregate


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
        "average": [],
        "task_arithmetic": [],
        "fisher_merging": [],
        "regmean_merging": [],
        "ties_merging": [],
        # "feddare" : ["p_drop"]
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

        if strategy in ["average", "task_arithmetic"]:
            return TaskArithmetic(server_model)
        elif strategy == "fisher_merging":
            return FisherMerging(server_model)
        elif strategy == "ties_merging":
            return TiesMerging(server_model)
        elif strategy == "regmean_merging":
            return RegmeanMerging(server_model)
        # elif strategy == "fedprox":
        #     # TODO: Double check when we have the args + kwargs (pass the params through argparse)
        #     mu = merging_kwargs.get("mu", 0.01)
        #     return FedProx(server_model, mu=mu)
        # elif strategy == "feddare":
        #     p_drop = merging_kwargs.get("p_drop", 0.5)
        #     return FedDare(server_model, p_drop=p_drop)
        else:
            raise ValueError(f"Unknown merging strategy: {strategy}")
