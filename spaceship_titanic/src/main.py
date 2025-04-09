from train_utils import *
import warnings
import time
def main():
    warnings.filterwarnings("ignore")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr=0.001
    train_data_path = "processed_data/train_data.csv"
    train_label_path = "processed_data/train_label.csv"
    test_data_path = "processed_data/test_data.csv"
    train_loader, test_loader= get_dataloader(train_data_path,train_label_path, test_data_path, batch_size=32)
    len_train = len(train_loader.dataset)
    train_loss,model=train_model( train_loader=train_loader, device=device,lr=lr, num_epochs=100)
    import matplotlib.pyplot as plt
    plt.plot(train_loss)
    plt.show()
    # val_model(model,device, val_loader )
    date=time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    output_path = f"predict/submission_dl_{date}.csv"
    test_model(model,device, test_loader,output_path )
    pass
if __name__=="__main__":
    main()