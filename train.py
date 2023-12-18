import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
import wandb
from adafnn import _l2


class CustomDataset(Dataset):
    def __init__(self, x_data_path="data/original_data.pt", y_data_path="data/original_data.pt", mode="train", val_size=0.3):
        assert mode in ("train", "val")
        self.mode = mode
        x_data = torch.load(x_data_path)
        y_data = torch.load(y_data_path)
        self.x_tr, self.x_val, self.y_tr, self.y_val = train_test_split(x_data, y_data, test_size=val_size,
                                                                        shuffle=True, random_state=123)

    def __len__(self):
        if self.mode == "train":
            return len(self.x_tr)
        else:
            return len(self.x_val)
    
    def __getitem__(self, index):
        if self.mode == "train":
            return self.x_tr[index], self.y_tr[index]
        else:
            return self.x_val[index], self.y_val[index]


def train(model, dataloader, loss_fn, optimizer, device):
    """
    train for 1 epoch
    """
    data_size = len(dataloader.dataset)
    ma_loss = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader, 1):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        # loss = loss_fn(y_pred, y) + model.R2()
        loss = _l2(y - y_pred, model.h)
        ma_loss += (loss.item() * len(y))  # bc. loss_fn predicts avg loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            print(f"Training loss: {loss.item():>7f} [{batch*len(y):>3d}/{data_size:>3d}]")
    ma_loss /= data_size    # moving average of loss over 1 epoch
    print(f"Train error:\n Avg loss: {ma_loss:>8f} \n")
    return ma_loss


def eval(model, dataloader, loss_fn, device):
    data_size = len(dataloader.dataset)
    loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            # loss += (loss_fn(y_pred, y).item() * len(y))
            loss = _l2(y - y_pred, model.h)
    loss /= data_size
    print(f"Validation/Test error:\n Avg loss: {loss:>8f} \n")
    return loss


def train_pipeline(model, config):
    with wandb.init(config=config, project="AdaFNN", group=f"{config['model_params']['n_base']}"):
        config = wandb.config
        train_dataset = CustomDataset(mode="train", x_data_path=config.x_data_path, y_data_path=config.y_data_path)
        train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True)
        val_dataset = CustomDataset(mode="val", x_data_path=config.x_data_path, y_data_path=config.y_data_path)
        val_dataloader = DataLoader(val_dataset, config.batch_size, shuffle=True)

        loss_fn = nn.MSELoss()
        optim = Adam(model.parameters(), lr=config.lr)
        scheduler = StepLR(optim, step_size=config.step_size, gamma=config.gamma, verbose=True)

        wandb.watch(model, loss_fn, log="all", log_freq=10)

        for n_epoch in range(1, config.epochs+1):
            print(f"\nEpoch: [{n_epoch} / {config.epochs}]")
            print("-"*30)

            train_loss = train(model, train_dataloader, loss_fn, optim, config.device)
            val_loss = eval(model, val_dataloader, loss_fn, config.device)

            scheduler.step()
            
            wandb.log({"train_loss":train_loss, "val_loss":val_loss}, step=n_epoch)