from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
def get_dataloader(train_data_path,train_label_path, val_data, batch_size=16):
    """
    Create DataLoader for training and validation datasets.

    Args:
        train_data : training data paths
        val_data : validation data paths
        batch_size (int): Batch size for DataLoader.

    Returns:
        tuple: DataLoader for training and validation datasets.
    """
    train_dataset = TrainDataset(train_data_path=train_data_path,train_label_path=train_label_path)
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
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).squeeze(1).long()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss  = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())
        if (epoch ) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            if loss.item()<min(train_loss) or epoch==0:
                save_model = model.state_dict()
                now_save_path = f"model/model.pth"
                torch.save(save_model, now_save_path)
    state_dict=torch.load(now_save_path)
    model.load_state_dict(state_dict)
    return train_loss,model
def test_model(model,device, val_loader,output_path):
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
            outputs = model(inputs).squeeze(1)
            predicted_labels = torch.argmax(outputs, dim=1)
            outputs_list.append(predicted_labels)
    outputs = torch.cat(outputs_list, dim=0)
    outputs = outputs.cpu().numpy()

    outputs = outputs.flatten()
    test_id=pd.read_csv("data/test.csv")["PassengerId"].values
    data = {
        "PassengerId": test_id,
        "Transported": outputs.astype(bool)
    }
    pd.DataFrame(data).to_csv(output_path, index=False)
def val_model(model,device, val_loader): 
    model.eval()
    # model.to(device)
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs,labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = (outputs > 0.5).int()
        acc = (predictions == labels).float().mean()
        print(f' Accuracy: {acc.item():.4f}')
class RegressionModel(nn.Module):
    def __init__(self,num_features):
        super(RegressionModel, self).__init__()
        self.num_features = num_features
        self.net=nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.TransformerEncoderLayer(d_model=128, nhead=8),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4, 2),
        )
    def forward(self,x):
        return self.net(x)
    


