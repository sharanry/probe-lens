import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score
from tqdm.autonotebook import tqdm


class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim=1, device="cpu", class_names=None):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, device=device)
        self.class_names = class_names

    def forward(self, x):
        return self.linear(x)

    def visualize_performance(
        self, dataloader: torch.utils.data.DataLoader, test=False
    ):
        preds = []
        gts = []
        for X, y in dataloader:
            pred = self(X)
            gt = y.argmax(dim=1)
            preds.append(pred.argmax(dim=1))
            gts.append(gt)
        preds = torch.cat(preds)
        gts = torch.cat(gts)

        accuracy = accuracy_score(gts.cpu(), preds.cpu())
        f2_score = fbeta_score(gts.cpu(), preds.cpu(), beta=2, average="weighted")
        cm = confusion_matrix(gts.cpu(), preds.cpu())
        _class_names = (
            self.class_names
            if self.class_names
            else [str(i) for i in range(cm.shape[0])]
        )
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=_class_names,
            yticklabels=_class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(
            f"Confusion Matrix (Accuracy: {accuracy:.4f}, F2 Score: {f2_score:.4f})"
        )
        if not test:
            plt.show()
        return plt

    def accuracy(self, dataloader: torch.utils.data.DataLoader):
        preds = []
        gts = []
        for X, y in dataloader:
            pred = self(X)
            gt = y.argmax(dim=1)
            preds.append(pred.argmax(dim=1))
            gts.append(gt)
        preds = torch.cat(preds)
        gts = torch.cat(gts)
        return (preds == gts).sum().item() / len(preds)

    def train_probe(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        val_dataloader: torch.utils.data.DataLoader | None = None,
        loss_fn: nn.Module = nn.BCEWithLogitsLoss(),
        epochs: int = 1000,
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
                if val_dataloader is not None:
                    dataset_names, datasets = (
                        ["train", "val"],
                        [dataloader, val_dataloader],
                    )
                else:
                    dataset_names, datasets = ["train"], [dataloader]
                if verbose and (epoch + 1) % 10 == 0:
                    losses = {}
                    for t, dataloader in zip(dataset_names, datasets):
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
