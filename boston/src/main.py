from train_utils import *
import warnings
def main():
    warnings.filterwarnings("ignore")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr=0.001
    train_data_path = "data/train.csv"
    test_data_path = "data/test.csv"
    train_loader, val_loader= get_dataloader(train_data_path, test_data_path, batch_size=32)
    
    train_loss,model=train_model( train_loader=train_loader, device=device,lr=lr, num_epochs=10)
    import matplotlib.pyplot as plt
    plt.plot(train_loss)
    plt.show()
    test_model(model,device, val_loader )
    pass
if __name__=="__main__":
    main()