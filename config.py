import os
import yaml
import argparse
from typing import Dict, Any, Optional, List
from pprint import pprint
from datetime import datetime


class ConfigManager:
    """Configuration manager for federated learning experiments."""
    
    DEFAULT_CONFIG_PATH = "configs"
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration directory.
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        os.makedirs(self.config_path, exist_ok=True)
        self.current_config = {}
        
    def create_default_configs(self):
        """Create default configuration files for different merging strategies."""
        base_config = {
            "gpu": "0",
            "dir": "runs",
            "name": "experiment",
            "save": "true",
            "dataset": "cifar10",
            "iid-alpha": 0.1,
            "clients": 100,
            "model": "vit_b_16",
            "seed": 0,
            "eval-freq": 10,
            "eval-first": "false",
            "eval-frac": 1.0,
            "eval-masked": "true",
            "server-opt": "adam",
            "server-lr": 1e-3,
            "server-batch": 10,
            "server-rounds": 100,
            "client-lr": 1e-3,
            "client-batch": 16,
            "client-epochs": 1,
            "client-freeze": "false",
            "server-freeze": "false",
            "freeze-a": "false",
            "lora-rank": 16,
            "lora-alpha": 16,
            "l2-clip-norm": 0.0,
            "noise-multiplier": 0.0,
            "use_wandb": True,
            "use_tensorboard": True,
            "project_name": "federated_merging",
            "entity": None,
            "merging_strategy": "fedavg"
        }
        
        self._save_config("base", base_config)
        
        strategies = {
            "fedavg": {},
            "fedprox": {"mu": 0.01},
            "fedmedian": {},
            "trimmedmean": {"trim_ratio": 0.1},
            "fednova": {},
            "fedper": {"personalized_layers": ["lora_A", "lora_B"]},
            "fedopt": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}
        }
        
        for strategy, params in strategies.items():
            config = base_config.copy()
            config["merging_strategy"] = strategy
            config.update(params)
            self._save_config(f"{strategy}_default", config)
            
        print(f"Created default configurations in '{self.config_path}' directory.")
    
    def _save_config(self, name: str, config: Dict[str, Any]):
        file_path = os.path.join(self.config_path, f"{name}.yaml")
        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def load_config(self, name: str) -> Dict[str, Any]:
        file_path = os.path.join(self.config_path, f"{name}.yaml")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration '{name}' not found at {file_path}")
            
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
            
        self.current_config = config
        return config
    
    def save_experiment_config(self, config: Dict[str, Any], name: Optional[str] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_prefix = name or config.get("name", "experiment")
        
        strategy = config.get("merging_strategy", "unknown")
        file_name = f"{name_prefix}_{strategy}_{timestamp}"
        
        file_path = os.path.join(self.config_path, f"{file_name}.yaml")
        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        return file_path
    
    def modify_config(self, **kwargs) -> Dict[str, Any]:
        self.current_config.update(kwargs)
        return self.current_config
    
    def print_config(self, config: Optional[Dict[str, Any]] = None):
        config = config or self.current_config
        print("\n--- Configuration ---")
        pprint(config)
        print("--------------------\n")
    
    def get_all_configs(self) -> List[str]:
        configs = []
        for file in os.listdir(self.config_path):
            if file.endswith(".yaml"):
                configs.append(os.path.splitext(file)[0])
        return configs

    def _get_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--config", type=str, help="Configuration name to load")
        parser.add_argument("--create-defaults", action="store_true", 
                            help="Create default configuration files")
        parser.add_argument("--list-configs", action="store_true", 
                            help="List all available configurations")
        
        parser.add_argument("--gpu", type=str, help="GPU device ID")
        parser.add_argument("--name", type=str, help="Experiment name")
        parser.add_argument("--dataset", type=str, help="Dataset name")
        parser.add_argument("--clients", type=int, help="Number of clients")
        parser.add_argument("--server-rounds", type=int, help="Number of server rounds")
        parser.add_argument("--server-batch", type=int, help="Server batch size")
        parser.add_argument("--client-lr", type=float, help="Client learning rate")
        parser.add_argument("--server-lr", type=float, help="Server learning rate")
        parser.add_argument("--seed", type=int, help="Random seed")
        
        parser.add_argument("--merging-strategy", type=str, 
                            choices=["fedavg", "fedprox", "fedmedian", "trimmedmean", 
                                    "fednova", "fedper", "fedopt"],
                            help="Merging strategy")
        
        parser.add_argument("--mu", type=float, help="Proximal term weight for FedProx")
        parser.add_argument("--trim-ratio", type=float, 
                            help="Trim ratio for TrimmedMean")
        parser.add_argument("--personalized-layers", type=str, nargs="+", 
                            help="Layer names to be treated as personalized in FedPer")
        
        return parser
    
    def process_cli_args(self) -> Dict[str, Any]:
        """
        Process command-line arguments and return configuration.
        
        Returns:
            Configuration dictionary.
        """
        parser = self._get_parser()
        args = parser.parse_args()
        
        if args.create_defaults:
            self.create_default_configs()
            exit(0)
            
        if args.list_configs:
            configs = self.get_all_configs()
            print("\nAvailable configurations:")
            for config in configs:
                print(f"  - {config}")
            print("")
            exit(0)
            
        if args.config:
            config = self.load_config(args.config)
        else:
            try:
                config = self.load_config("base")
            except FileNotFoundError:
                print("No configuration specified and no 'base' config found.")
                print("Creating default configurations...")
                self.create_default_configs()
                config = self.load_config("base")
        
        for arg_name, arg_value in vars(args).items():
            if arg_value is not None and arg_name not in ["config", "create_defaults", "list_configs"]:
                config[arg_name.replace("_", "-")] = arg_value
        
        self.current_config = config
        return config


def cli_run():
    """Entry point for command-line use."""
    config_manager = ConfigManager()
    
    config = config_manager.process_cli_args()
    config_manager.print_config(config)
    config_path = config_manager.save_experiment_config(config)
    print(f"Saved experiment configuration to {config_path}")
    
    return config


if __name__ == "__main__":
    cli_run()