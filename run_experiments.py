import subprocess
import argparse
import yaml
from datetime import datetime
from config import ConfigManager

def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple federated learning experiments")
    parser.add_argument("--config-list", type=str, help="YAML file with list of configurations to run")
    parser.add_argument("--strategies", type=str, nargs="+", default=None, 
                        help="List of merging strategies to run (overrides config-list)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to use (overrides config)")
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds (overrides config)")
    parser.add_argument("--clients", type=int, default=None, help="Number of clients (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--gpu", type=str, default=None, help="GPU device ID (overrides config)")
    parser.add_argument("--timestamp", action="store_true", help="Add timestamp to experiment name")
    parser.add_argument("--experiment-name", type=str, default=None, help="Base name for experiments")
    return parser.parse_args()

def load_config_list(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def run_experiment(config_name, overrides=None):
    cmd = ["python", "train_sparse_lora_vanilla.py", "--config", config_name]
    
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
    
    print(f"\n--- Running: {' '.join(cmd)} ---\n")
    
    subprocess.run(cmd)

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if args.timestamp else None
    
    overrides = {
        "dataset": args.dataset,
        "server-rounds": args.rounds,
        "clients": args.clients,
        "seed": args.seed,
        "gpu": args.gpu,
    }
    
    overrides = {k: v for k, v in overrides.items() if v is not None}
    
    if args.experiment_name:
        overrides["name"] = args.experiment_name
        if timestamp:
            overrides["name"] = f"{overrides['name']}_{timestamp}"
    
    elif args.config_list:
        # Run multiple experiments from config list
        config_list = load_config_list(args.config_list)
        
        for config_item in config_list:
            if isinstance(config_item, str):
                run_experiment(config_item, overrides)
            elif isinstance(config_item, dict):
                config_name = config_item.pop("config", "fedavg")
                
                item_overrides = overrides.copy()
                item_overrides.update(config_item)
                
                run_experiment(config_name, item_overrides)
                
    else:
        # Run default experiment
        run_experiment("fedavg", overrides)

if __name__ == "__main__":
    main()