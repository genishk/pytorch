class Config:
    batch_size = 64
    epochs = 10
    learning_rate = 0.01
    momentum = 0.5
    cuda = True
    seed = 42
    log_interval = 100
    save_model = True
    experiment_name = 'mnist_training'
    checkpoint_dir = 'checkpoints' 