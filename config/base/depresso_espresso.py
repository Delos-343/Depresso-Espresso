config = {
    'distributed': False,
    'gpu': 0,
    'local_rank': 0,
    'nodes': 1,
    'use_slurm': False,
    'world_size': 1,
    'learning_rate': 1e-4,
    'epochs': 20,              # Increased epochs for fine-tuning
    'batch_size': 16,
    'data_dir': 'data',
    'model_dir': 'model',
    'use_pretrained': True,    # Use pre-trained ResNetTransfer model
    'freeze_layers': False,    # Unfreeze all layers for fine-tuning
    'evaluate': False,
    'patience': 5              # For early stopping (optional)
}
