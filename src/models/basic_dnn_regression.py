import torch
import torch.nn as nn


class BasicDNNRegression(nn.Module):
    def __init__(self):
        super(BasicDNNRegression, self).__init__()
