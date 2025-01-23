import torch
import torch.nn as nn
from ai.Train import transform_to_tensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    return torch.load(model_path)


def predict_single(model, seq, device=DEVICE):
    criterion = nn.MSELoss(reduction='sum').to(device)
    with torch.no_grad():
        seq = seq.reshape(1, -1)
        # print(seq.shape)
        seq = transform_to_tensor(seq)
        seq = seq.to(device)
        pre_seq = model(seq)
        loss = criterion(seq, pre_seq)
        prediction = pre_seq.cpu().numpy()[0]
        loss = loss.cpu().numpy()
    return prediction, loss


def predict_dataset(model, test_data, device=DEVICE):
    predictions, losses = [], []
    for seq in test_data:
        prediction, loss = predict_single(model, seq, device)
    return predictions, losses
