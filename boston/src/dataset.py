import torch
import pandas as pd
import csv
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path,delim_whitespace=True,header=None)
        self.train_data = self.data.iloc[:, :-1].to_numpy()
        self.train_data = torch.tensor(self.train_data, dtype=torch.float32)
        self.train_labels =  self.data.iloc[:, -1].to_numpy()
        self.train_labels = torch.tensor(self.train_labels, dtype=torch.float32)
        self.feature_num = self.train_data.shape[1]

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        return self.train_data[idx], self.train_labels[idx]

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # Load the dataset from the specified path
        self.data = pd.read_csv(data_path,delim_whitespace=True,header=None)
        # Convert the DataFrame to a PyTorch tensor
        self.test_data = self.data.to_numpy()
        self.test_data = torch.tensor(self.test_data, dtype=torch.float32)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.test_data[idx]