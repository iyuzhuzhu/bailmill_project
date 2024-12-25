import torch
import torch.nn as nn
import numpy as np


def train_model(model, train_dataset, val_dataset, n_epochs, device, path, lr=5e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum').to(device)
    #     criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        for step, (seq_true,) in enumerate(train_dataset):
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        history['train'].append(np.mean(train_losses))  # 记录训练epoch损失
        if epoch % 20 == 0:
            print("第{}个train_epoch，loss：{}".format(epoch, history['train'][-1]))
        torch.save(model, path)
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.reshape(1, -1)
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['val'].append(val_loss)
    return history


def main():
    pass


if __name__ == '__main__':
    pass
