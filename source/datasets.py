from pandas import DataFrame
from torch import device, float32, tensor, uint8
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, features: DataFrame, targets: DataFrame, device: device):
        self.features = tensor(
            features, dtype=float32, device=device, requires_grad=True
        )
        self.targets = tensor(targets, dtype=float32, device=device, requires_grad=True)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx, :], self.targets[idx]
