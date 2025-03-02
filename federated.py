import torch
from torch import nn


class BaseFederated:
    def __init__(self, server_model: nn.Module, lr: float = 0.01):
        self.server_model = server_model
        self.lr = lr
        self.server_params = {n: p for n, p in server_model.named_parameters() if p.requires_grad}
        self.round_idx = 0
        
    def init_state(self):
        pass
        
    def step(self, merged_update: dict[str, torch.Tensor], sample_num_list: list[int] | None = None):
        raise NotImplementedError()
    
    def increment_round(self):
        self.round_idx += 1

class FedSGD(BaseFederated):
    def step(self, merged_update: dict[str, torch.Tensor], sample_num_list: list[int] | None = None):
        for key in self.server_params.keys():
            if key in merged_update:
                self.server_params[key].data -= self.lr * merged_update[key]


class FedAvgM(BaseFederated):
    def __init__(self, server_model: nn.Module, lr: float = 0.01, beta1: float = 0.9):
        super().__init__(server_model, lr)
        self.beta1 = beta1
        self.momentum = None
        
    def init_state(self):
        self.momentum = {n: torch.zeros_like(p.data) for n, p in self.server_params.items()}
        
    def step(self, merged_update: dict[str, torch.Tensor], sample_num_list: list[int] | None = None):
        if self.momentum is None:
            self.init_state()
            
        for key in self.server_params.keys():
            if key in merged_update:
                
                if self.round_idx > 0:
                    self.momentum[key] = self.beta1 * self.momentum[key] + (1 - self.beta1) * merged_update[key]
                else:
                    self.momentum[key] = merged_update[key]
                    
                
                self.server_params[key].data -= self.momentum[key]


class FedAdaGrad(BaseFederated):
    def __init__(self, server_model: nn.Module, lr: float = 0.01, tau: float = 1e-8):
        super().__init__(server_model, lr)
        self.tau = tau
        self.sum_squared_grads = None
        
    def init_state(self):
        self.sum_squared_grads = {n: torch.zeros_like(p.data) for n, p in self.server_params.items()}
        
    def step(self, merged_update: dict[str, torch.Tensor], sample_num_list: list[int] | None = None):
        if self.sum_squared_grads is None:
            self.init_state()
            
        for key in self.server_params.keys():
            if key in merged_update:
                self.sum_squared_grads[key] += torch.square(merged_update[key])
                
                self.server_params[key].data -= self.lr * torch.div(
                    merged_update[key], 
                    torch.sqrt(self.sum_squared_grads[key]) + self.tau
                )


class FedAdam(BaseFederated):
    def __init__(self, server_model: nn.Module, lr: float = 0.01, 
                 beta1: float = 0.9, beta2: float = 0.999, tau: float = 1e-8):
        super().__init__(server_model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m = None  
        self.v = None  
        
    def init_state(self):
        self.m = {n: torch.zeros_like(p.data) for n, p in self.server_params.items()}
        self.v = {n: torch.zeros_like(p.data) for n, p in self.server_params.items()}
        
    def step(self, merged_update: dict[str, torch.Tensor], sample_num_list: list[int] | None = None):
        if self.m is None or self.v is None:
            self.init_state()
            
        for key in self.server_params.keys():
            if key in merged_update:
                
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * merged_update[key]
                
                
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * torch.square(merged_update[key])
                
                
                m_hat = self.m[key] / (1 - self.beta1 ** (self.round_idx + 1))
                v_hat = self.v[key] / (1 - self.beta2 ** (self.round_idx + 1))
                
                
                self.server_params[key].data -= self.lr * torch.div(m_hat, torch.sqrt(v_hat) + self.tau)


class FedYogi(BaseFederated):
    def __init__(self, server_model: nn.Module, lr: float = 0.01, 
                 beta1: float = 0.9, beta2: float = 0.999, tau: float = 1e-8):
        super().__init__(server_model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m = None  
        self.v = None  
        
    def init_state(self):
        self.m = {n: torch.zeros_like(p.data) for n, p in self.server_params.items()}
        self.v = {n: torch.zeros_like(p.data) for n, p in self.server_params.items()}
        
    def step(self, merged_update: dict[str, torch.Tensor], sample_num_list: list[int] | None = None):
        if self.m is None or self.v is None:
            self.init_state()
            
        for key in self.server_params.keys():
            if key in merged_update:
                
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * merged_update[key]
                
                
                delta_squared = torch.square(merged_update[key])
                self.v[key] = self.v[key] - (1 - self.beta2) * delta_squared * torch.sign(self.v[key] - delta_squared)
                
                
                m_hat = self.m[key] / (1 - self.beta1 ** (self.round_idx + 1))
                v_hat = self.v[key] / (1 - self.beta2 ** (self.round_idx + 1))
                
                
                self.server_params[key].data -= self.lr * torch.div(m_hat, torch.sqrt(v_hat) + self.tau)


class SCAFFOLD(BaseFederated):
    def __init__(self, server_model: nn.Module, lr: float = 0.01, num_clients: int = 100):
        super().__init__(server_model, lr)
        self.num_clients = num_clients
        self.global_auxiliary = None
        self.auxiliary_model_list = None
        
    def init_state(self, client_ids=None):
        self.global_auxiliary = {n: torch.zeros_like(p.data) for n, p in self.server_params.items()}
        
        if self.auxiliary_model_list is None:
            self.auxiliary_model_list = []
            for _ in range(self.num_clients):
                self.auxiliary_model_list.append({n: torch.zeros_like(p.data) for n, p in self.server_params.items()})
        
    def step(self, merged_update: dict[str, torch.Tensor], auxiliary_deltas: list[dict[str, torch.Tensor]] = None, 
             client_ids: list[int] = None, sample_num_list: list[int | None] | None = None):
        if self.global_auxiliary is None:
            self.init_state()
            
        for key in self.server_params.keys():
            if key in merged_update:
                self.server_params[key].data -= self.lr * merged_update[key]
        
        if auxiliary_deltas and client_ids:
            for key in self.global_auxiliary.keys():
                delta_auxiliary = sum([auxiliary_deltas[i][key] for i in range(len(auxiliary_deltas))])
                self.global_auxiliary[key] += delta_auxiliary / len(auxiliary_deltas)


class FederatedFactory:
    @staticmethod
    def get_federated_optimizer(optimizer_name: str, server_model: nn.Module, args=None, **kwargs) -> BaseFederated:
        optimizer_name = optimizer_name.lower()
        
        lr = kwargs.get('lr', 0.01)
        
        if args is not None:
            args_dict = vars(args) if hasattr(args, "__dict__") else args
            server_hparams = args_dict.get('server_hparams', {})
            
            if server_hparams:
                beta1 = server_hparams.get('beta1', 0.9)
                beta2 = server_hparams.get('beta2', 0.999)
                tau = server_hparams.get('tau', 1e-8)
                
                kwargs['beta1'] = kwargs.get('beta1', beta1)
                kwargs['beta2'] = kwargs.get('beta2', beta2)
                kwargs['tau'] = kwargs.get('tau', tau)
        
        if optimizer_name == "fedsgd" or optimizer_name == "fedavg":
            return FedSGD(server_model, lr=lr)
            
        elif optimizer_name == "fedavgm":
            beta1 = kwargs.get('beta1', 0.9)
            return FedAvgM(server_model, lr=lr, beta1=beta1)
            
        elif optimizer_name == "fedadagrad":
            tau = kwargs.get('tau', 1e-8)
            return FedAdaGrad(server_model, lr=lr, tau=tau)
            
        elif optimizer_name == "fedadam":
            beta1 = kwargs.get('beta1', 0.9)
            beta2 = kwargs.get('beta2', 0.999)
            tau = kwargs.get('tau', 1e-8)
            return FedAdam(server_model, lr=lr, beta1=beta1, beta2=beta2, tau=tau)
            
        elif optimizer_name == "fedyogi":
            beta1 = kwargs.get('beta1', 0.9)
            beta2 = kwargs.get('beta2', 0.999)
            tau = kwargs.get('tau', 1e-8)
            return FedYogi(server_model, lr=lr, beta1=beta1, beta2=beta2, tau=tau)
            
        elif optimizer_name == "scaffold":
            num_clients = kwargs.get('num_clients', 100)
            return SCAFFOLD(server_model, lr=lr, num_clients=num_clients)
            
        else:
            raise ValueError(f"Unknown federated optimizer: {optimizer_name}")