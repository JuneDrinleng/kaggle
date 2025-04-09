from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
def get_dataloader(train_data, val_data, batch_size=16):
    """
    Create DataLoader for training and validation datasets.

    Args:
        train_data : training data paths
        val_data : validation data paths
        batch_size (int): Batch size for DataLoader.

    Returns:
        tuple: DataLoader for training and validation datasets.
    """
    train_dataset = TrainDataset(train_data)
    val_dataset = TestDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train_model( train_loader,device,lr, num_epochs=10):
    """
    Train the model.

    Args:
        model : PyTorch model to be trained
        train_loader : DataLoader for training data
        criterion : loss function
        optimizer : optimizer for training
        num_epochs (int): Number of epochs to train the model.

    Returns:
        None
    """
    num_features = train_loader.dataset.feature_num
    model = RegressionModel(num_features)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = RMSE(outputs, labels)
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())
        if (epoch ) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            save_model = model.state_dict()
            now_save_path = f"model/model_epoch_{epoch + 1}.pth"
            torch.save(save_model, now_save_path)
    return train_loss,model
def test_model(model,device, val_loader):
    """
    Evaluate the model on validation data.
    Args:
        model : PyTorch model to be evaluated
        val_loader : DataLoader for validation data
        criterion : loss function
    Returns:
        float: Average loss on validation data.
    """   
    model.eval()
    # model.to(device)
    outputs_list = []
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs_list.append(outputs)
    outputs = torch.cat(outputs_list, dim=0)
    outputs = outputs.cpu().numpy()
    outputs = outputs.flatten()
    data = {
        "id": np.arange(len(outputs)),
        "predict": outputs  
    }
    pd.DataFrame(data).to_csv("predict/outputs.csv", index=False)

class RegressionModel(nn.Module):
    def __init__(self,num_features):
        super(RegressionModel, self).__init__()
        self.num_features = num_features
        self.net=nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self,x):
        return self.net(x)
    

def RMSE(y_pred,y_true):
    mse_loss = nn.MSELoss()
    rmse_loss = torch.sqrt(mse_loss(y_pred, y_true))
    return rmse_loss
