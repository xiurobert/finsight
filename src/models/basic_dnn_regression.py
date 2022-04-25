import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm


class BasicDNNRegression(nn.Module):
    def __init__(self):
        super(BasicDNNRegression, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        return x

    def train_one_epoch(self,
                        train_loader: torch.utils.data.DataLoader,
                        optimizer: torch.optim.Optimizer,
                        criterion: torch.nn.Module,
                        epoch_number=0):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(dev)
        self.train()
        loss_tally = 0
        for i, (features, labels) in enumerate((prog_bar := tqdm(train_loader, total=len(train_loader)))):
            features, labels = features.to(dev), labels.to(dev)
            # labels = labels.view(-1, 1)

            predictions = self(features)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_tally += loss.item()
            prog_bar.set_description(f"Epoch: {epoch_number + 1}, "
                                     f"Batch #: {i+1}, "
                                     f"Current Loss: {loss.item():.3f}, "
                                     f"Avg Loss: {loss_tally / (i + 1):.3f}")
