import torch
import torch.nn as nn

class BasePolicy(object):
    def __init__(self, config):
        super(BasePolicy, self).__init__()
        self.config = config

    def train(self):
        raise NotImplementedError
    
    def evaluate(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError
