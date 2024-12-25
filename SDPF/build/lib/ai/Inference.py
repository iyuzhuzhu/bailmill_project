import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    return torch.load(model_path)


def predict(model, test_data, device=DEVICE):
    predictions, losses = [], []
    criterion = nn.MSELoss(reduction='sum').to(device)
    with torch.no_grad():
        for seq in test_data:
            seq = seq.reshape(1, -1)
            seq = seq.to(device)
            pre_seq = model(seq)
            loss = criterion(seq, pre_seq)
            predictions.append(pre_seq.cpu().numpy()[0])
            losses.append(loss.item())
    return predictions, losses
