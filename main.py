import argparse
from argparse import Namespace
import importlib.util
from data.runner import Runner

def load_config(config_path):

    """
    Dynamically load the configuration from a Python file.
    The file must define a 'config' dictionary.
    """

    spec = importlib.util.spec_from_file_location("depresso_espresso", config_path)
    
    config_module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(config_module)

    return config_module.config


def train_depresso():
    args = Namespace(
        config="config/base/depresso-espresso.py",
        distributed=False,
        gpu=0,
        local_rank=0,
        nodes=1,
        use_slurm=False,
        world_size=1,
        cfg_options=None  # You can add a dictionary of additional training options here
    )
    
    # Load configuration from the specified file
    cfg = load_config(args.config)
    
    # Merge Namespace values into the config dictionary
    for key, value in vars(args).items():
        cfg[key] = value
    
    # Instantiate the Runner with the merged configuration
    runner = Runner(cfg)
    
    # Run training
    runner.train()


def eval_depresso():
    args = Namespace(
        config="config/base/depresso-espresso.py",
        distributed=False,
        gpu=0,
        local_rank=0,
        nodes=1,
        use_slurm=False,
        world_size=1,
        cfg_options={'evaluate': True}  # Add evaluation-specific options here
    )
    
    # Load configuration from the specified file
    cfg = load_config(args.config)
    
    # Merge Namespace values into the config dictionary
    for key, value in vars(args).items():
        cfg[key] = value
    
    # Instantiate the Runner with the merged configuration
    runner = Runner(cfg)
    
    # Run evaluation
    runner.eval()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Depresso-Espresso Runner")

    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help="Mode to run the project: 'train' for training, 'eval' for evaluation.")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_depresso()
    elif args.mode == 'eval':
        eval_depresso()
