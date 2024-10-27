from tqdm import tqdm
import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim=1, device="cpu"):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, device=device)

    def forward(self, x):
        return self.linear(x)

    def train_probe(
        self,
        dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        verbose: bool = True,
    ):
        tqdm_epochs = tqdm(range(epochs), desc="Training Probe", unit="epoch")
        for epoch in tqdm_epochs:
            loss_sum = 0
            for X, y in dataloader:
                optimizer.zero_grad()
                pred = self(X)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                if verbose and (epoch + 1) % 10 == 0:
                    losses = {}
                    for t, dataloader in zip(
                        ["train", "val"], [dataloader, val_dataloader]
                    ):
                        loss_sum = 0
                        correct = 0
                        size = 0
                        for X, y in dataloader:
                            pred = self(X)
                            argmax_pred = pred.argmax(dim=1)
                            gt = y.argmax(dim=1)
                            correct += (argmax_pred == gt).sum().item()
                            loss = loss_fn(pred, y)
                            loss_sum += loss.item()
                            size += X.size(0)
                        losses[t + "_loss"] = loss_sum / size
                        losses[t + "_acc"] = correct / size
                    tqdm_epochs.set_postfix(**losses)
        return
