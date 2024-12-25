import os.path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from ai.ae_model import LstmAutoencoder


EPOCH = 150
LR = 0.0004
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_SIZE = 0.1
BATCH_SIZE = 50


def transform_to_tensor(data):
    return torch.Tensor(data)


def split_train_val_test(data, test_size, val_test_size, random):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random)
    val_data, test_data = train_test_split(test_data, test_size=val_test_size, random_state=random)
    return train_data, val_data, test_data


def split_data_batch(data, batch_size):
    train_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    # for train_batch in train_loader:
    #     print(train_batch[0].shape)
    #     break
    return train_loader


def preprocessing_training_data(training_data, test_size=TEST_SIZE, val_test_size=0.5, random=RANDOM_SEED,
                                batch_size=BATCH_SIZE):
    """
    对数据转化为Tensor，进行预处理，划分训练集，测试集，试验集，以及将训练集划分batch，参数为train模块的全局变量设置
    """
    training_data = transform_to_tensor(training_data)
    train_data, val_data, test_data = split_train_val_test(training_data, test_size, val_test_size, random)
    train_data = split_data_batch(train_data, batch_size)
    return train_data, val_data, test_data


def create_train_model(train_data, val_data, model_path, folder_path=None, epochs=EPOCH, device=DEVICE, lr=LR,
                       is_plot_loss=True, loss_picture_name='history.png'):
    model = LstmAutoencoder()
    model = model.to(device)
    history = train_model(model, train_data, val_data, model_path, epochs, device, lr)
    if is_plot_loss:
        try:
            plot_history(history, folder_path, loss_picture_name)
        except Exception as e:
            print(e)


def train_model(model, train_dataset, val_dataset, path, n_epochs, device, lr):
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


def plot_history(history, save_folder_path, fig_name):
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    history['train'] = np.array(history['train'])
    plt.plot(history['train'] / 50)
    # ax.plot(history['val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.legend(['train', 'test'])
    plt.title('Loss over training epochs')
    plt.subplot(122)
    plt.plot(history['val'])
    # ax.plot(history['val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.legend(['train', 'test'])
    plt.title('Loss over val epochs')
    save_path = os.path.join(save_folder_path, fig_name)
    plt.savefig(save_path)
    plt.close()


def main():
    pass


if __name__ == '__main__':
    pass
