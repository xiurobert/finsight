import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm


class ConvTimeSeries(nn.Module):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super(ConvTimeSeries, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
        )