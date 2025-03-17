config = {
    'distributed': False,
    'gpu': 0,
    'local_rank': 0,
    'nodes': 1,
    'use_slurm': False,
    'world_size': 1,
    'learning_rate': 1e-4,       # Lower learning rate for more stable training
    'epochs': 10,
    'batch_size': 16,            # Adjust based on your GPU memory
    'data_dir': 'data',
    'model_dir': 'model',
    'use_pretrained': True,      # Use pre-trained ResNetTransfer model for better feature extraction
    'evaluate': False,           # Flag to indicate evaluation mode
    'patience': 100              # Patience is not used here (training always runs full epochs)
}
