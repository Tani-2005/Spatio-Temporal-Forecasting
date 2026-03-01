import torch
from torch.utils.data import Dataset
import numpy as np

class EpidemicDataset(Dataset):
    """
    A custom PyTorch Dataset for Spatio-Temporal Epidemic Forecasting.
    Transforms continuous time-series data into overlapping sliding windows.
    """
    def __init__(self, features, targets, seq_len=12, pred_len=4):
        """
        Args:
            features (np.ndarray): 3D array of shape (Total_Time_Steps, Num_Nodes, Num_Features)
                                   e.g., (520 weeks, 2 cities, 3 weather features)
            targets (np.ndarray): 2D array of shape (Total_Time_Steps, Num_Nodes)
                                  e.g., (520 weeks, 2 cities' dengue cases)
            seq_len (int): How many past weeks to look at (History / M).
            pred_len (int): How many future weeks to predict (Horizon / H).
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Calculate how many valid windows we can extract
        self.total_windows = len(self.features) - self.seq_len - self.pred_len + 1

    def __len__(self):
        # PyTorch needs to know exactly how many samples exist in the dataset
        return self.total_windows

    def __getitem__(self, idx):
        """
        Fetches one window of history (X) and its corresponding future targets (Y).
        """
        # The historical window: from idx to (idx + seq_len)
        x_window = self.features[idx : idx + self.seq_len]
        
        # The future target window: from (idx + seq_len) to (idx + seq_len + pred_len)
        y_window = self.targets[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        
        return x_window, y_window

def create_dataloaders(dataset, batch_size=32, train_split=0.8):
    """
    Splits the dataset into Training and Validation sets, and wraps them in DataLoaders.
    """
    from torch.utils.data import DataLoader, random_split
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # We use sequential splitting for time-series, not random_split, 
    # to prevent data leakage from the future into the past!
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader