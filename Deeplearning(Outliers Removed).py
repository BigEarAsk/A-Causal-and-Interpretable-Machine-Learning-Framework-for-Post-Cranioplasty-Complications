import torch
import torch.nn as nn
import numpy as np
import xlrd
import random
from openpyxl import load_workbook

# define autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, encoding_dim))
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, input_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_data(x_index = [0,13,15,22],path = None): # get data
    x = []
    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(0)
    for i in range(1,sheet.nrows):  
        user = sheet.row_values(i)
        user = np.array(user)
        x.append(user[x_index])
    return np.array(x)

def set_seed(seed=100):  # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def del_ws_rows_cols(rowd,path = None,name=None):  # delete data and save updated data

    wb = load_workbook(path)
    ws = wb['Sheet1']   
    rowd = sorted(rowd, reverse=True)  
    
    for r in rowd:                     
        ws.delete_rows(r)
    
    wb.save('./Outlier/Outlier_Removed/'+name+'.xlsx')  


if __name__ == '__main__':
    set_seed()

    paths = ['xxxx.xlsx']   # input data(.xlsx) path

    names = ['xxxx']        # output file name (without suffix)

    for path,name in zip(paths,names):
        X_train = get_data(path = path)
        # transform the data to tensor
        X_train_tensor = torch.tensor(X_train,dtype=torch.float)

        # create and train autoencoder model
        input_dim = X_train_tensor.shape[1]
        encoding_dim = 10  
        autoencoder_model = Autoencoder(input_dim, encoding_dim)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=0.001)

        num_epochs = 100
        batch_size = 32

        for epoch in range(num_epochs):
            for i in range(0, X_train_tensor.size(0), batch_size):
                input_data = X_train_tensor[i:i+batch_size]
                # forward
                output_data = autoencoder_model(input_data)
                loss = criterion(output_data, input_data)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # reconstruct the data by trained model
        with torch.no_grad():
            reconstructed_data = autoencoder_model(X_train_tensor).numpy()

        # compute construction error (mse)
        mse = np.mean(np.square(X_train - reconstructed_data), axis=1)

        # set the threshold value to identify the outliers
        threshold = np.mean(mse) + 2 * np.std(mse)

        # print indexes of outliers 
        outlier_indices = np.where(mse > threshold)[0]
        print("Outlier Index：", outlier_indices)  

        del_ws_rows_cols(outlier_indices+2,path = path,name=name)
