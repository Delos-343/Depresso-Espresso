config = {
    'distributed': False,
    'gpu': 0,
    'local_rank': 0,
    'nodes': 1,
    'use_slurm': False,
    'world_size': 1,
    'learning_rate': 1e-4,
    'epochs': 30,              # Increase epochs for full fine-tuning
    'batch_size': 16,
    'data_dir': 'data',
    'model_dir': 'model',
    'use_pretrained': True,
    'freeze_layers': True,
    'use_focal_loss': True,    # Use Focal Loss to focus on hard examples
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'evaluate': False,
    'patience': 5
}
