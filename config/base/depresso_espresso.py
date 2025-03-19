config = {
    'distributed': False,
    'gpu': 0,
    'local_rank': 0,
    'nodes': 1,
    'use_slurm': False,
    'world_size': 1,
    'learning_rate': 1e-4,       # Lower learning rate for stable fine-tuning
    'epochs': 25,                # Increase epochs for fine-tuning
    'batch_size': 16,            # Adjust based on your GPU memory
    'data_dir': 'data',
    'model_dir': 'model',
    'use_pretrained': True,      # Use pre-trained ResNetTransfer model
    'freeze_layers': False,      # Unfreeze all layers for fine-tuning
    'use_focal_loss': True,      # Set True to use Focal Loss instead of CrossEntropyLoss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'evaluate': False,
    'patience': 5                # Patience for early stopping (if desired)
}
