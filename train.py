import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import wandb


class CustomDataset(Dataset):
    def __init__(self, x_data_path="data/mydata/train/noise_train.pt", y_data_path="data/mydata/train/noise_train.pt"):
        self.x = torch.load(x_data_path)
        self.y = torch.load(y_data_path)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


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
        loss = loss_fn(y_pred, y) + model.R2()
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
            loss += (loss_fn(y_pred, y).item() * len(y))
    loss /= data_size
    print(f"Validation/Test error:\n Avg loss: {loss:>8f} \n")
    return loss


def train_pipeline(model, config):
    with wandb.init(config=config, project="AdaFNN", group=f"{config['model_params']['n_base']}"):
        config = wandb.config
        train_dataset = CustomDataset(x_data_path=config.x_train_path, y_data_path=config.y_train_path)
        train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True)
        val_dataset = CustomDataset(x_data_path=config.x_val_path, y_data_path=config.y_val_path)
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