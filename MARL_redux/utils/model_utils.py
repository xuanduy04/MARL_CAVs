from torch.optim import Adam, AdamW

def get_optimizer(params, lr, optimizer = 'adam'):
    if optimizer == 'sgd':
        optimizer = SGD(params, lr=lr)
    elif optimizer == 'adam':
        optimizer = Adam(params, lr=lr)
    elif optimizer == 'adamW':
        optim = AdamW(params, lr=lr)
    return optimizer