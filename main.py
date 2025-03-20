import argparse
from argparse import Namespace
import importlib.util
from experiments.runner import Runner

def load_config(config_path):
    
    """
    Dynamically load the configuration from a Python file.
    The file must define a 'config' dictionary.
    """
    
    spec = importlib.util.spec_from_file_location("depresso_config", config_path)
    
    config_module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(config_module)
    
    return config_module.config


def train():
    
    args = Namespace(
        config="config/base/depresso_espresso.py",
        distributed=False,
        gpu=0,
        local_rank=0,
        nodes=1,
        use_slurm=False,
        world_size=1,
        cfg_options=None  # You can add training-specific options here
    )
    
    # Load configuration from the specified file
    cfg = load_config(args.config)
    
    # Merge Namespace values into the config dictionary
    for key, value in vars(args).items():
        cfg[key] = value
    
    runner_instance = Runner(cfg)

    runner_instance.train()


def eval():
    
    args = Namespace(
        config="config/base/depresso_espresso.py",
        distributed=False,
        gpu=0,
        local_rank=0,
        nodes=1,
        use_slurm=False,
        world_size=1,
        cfg_options={'evaluate': True}  # Add evaluation-specific options here
    )
    
    cfg = load_config(args.config)
    
    for key, value in vars(args).items():
        cfg[key] = value
    
    runner_instance = Runner(cfg)

    runner_instance.eval()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Depresso-Espresso Runner")
    
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help="Mode to run the project: 'train' for training, 'eval' for evaluation.")
    
    args = parser.parse_args()
    
    # python main.py --mode train
    if args.mode == 'train':
        train()
    
    # python main.py --mode eval
    elif args.mode == 'eval':
        eval()
