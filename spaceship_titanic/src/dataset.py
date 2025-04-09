import torch
import pandas as pd
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, train_data_path,train_label_path):
        self.train_data = pd.read_csv(train_data_path)
        self.train_data = self.train_data.to_numpy()
        self.train_data = torch.tensor(self.train_data, dtype=torch.float32)
        self.train_labels =  pd.read_csv(train_label_path)
        self.train_labels = self.train_labels.to_numpy()
        self.train_labels = torch.tensor(self.train_labels, dtype=torch.float32)
        self.feature_num = self.train_data.shape[1]
        means = self.train_data.mean()
        stds = self.train_data.std()
        stds[stds == 0] = 1 
        self.train_data = (self.train_data - means) / stds

    def __len__(self):
        return len(self.train_data)
    

    def __getitem__(self, idx):
        return self.train_data[idx], self.train_labels[idx]

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.test_data = self.data.to_numpy()
        self.test_data = torch.tensor(self.test_data, dtype=torch.float32)
        means = self.test_data.mean()
        stds = self.test_data.std()
        stds[stds == 0] = 1
        self.test_data = (self.test_data - means) / stds
        self.feature_num = self.test_data.shape[1]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.test_data[idx]