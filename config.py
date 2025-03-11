import os
import yaml
import argparse
from typing import Dict, Any, Optional


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
        self.current_config = {}
        
    def load_config(self, name: str) -> Dict[str, Any]:
        file_path = os.path.join(self.config_path, f"{name}.yaml")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration '{name}' not found at {file_path}")
            
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
            
        self.current_config = config
        return config
    
    def _get_parser(self) -> argparse.ArgumentParser:
        """
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
        """
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
                                    "fednova", "fedper", "fedavgm", "fedadam", "fedadagrad",
                                    "scaffold"],
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
            self.create_default_configs()  # TODO: Implement this method
            exit(0)
            
        if args.list_configs:
            configs = self.get_all_configs()  # TODO: Implement this method
            print("\nAvailable configurations:")
            for config in configs:
                print(f"  - {config}")
            print("")
            exit(0)
        
        if args.config:
            config = self.load_config(args.config)
        else:
            config = self.load_config("fedavg")

        config = {k.replace("-", "_"): v for k, v in config.items()}  # Replace hyphens with underscores

        print("Loaded configuration:")
        for k, v in config.items():
            print(f"  - {k}: {v}")
        
        # Override or add method-specific configuration with command-line arguments
        for arg_name, arg_value in vars(args).items():
            if arg_value is not None and arg_name not in ["config", "create_defaults", "list_configs"]:
                config[arg_name.replace("-", "_")] = arg_value
        
        self.current_config = config
        
        return config
