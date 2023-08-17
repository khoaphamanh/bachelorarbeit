from helper import DataPreprocessing
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler
from timeit import default_timer

start = default_timer()

#global variable
window_size = 25600
end_of_life = {11:28020,12:8700,21:9100,22:7960,31:5140,32:16360,13:23740,14:14270,15:24620,16:24470,17:22580,23:19540,24:7500,25:23100,26:7000,27:2290,33:4330}

#function to calculate the scaler: 0:dead, label = eol 1:healthy, label = begin
def scaler (bearing):
    eol = end_of_life[bearing]
    value = np.array([0,eol]).reshape(-1,1)
    min_max_scaler = MinMaxScaler(feature_range=(0,1))
    scaler = min_max_scaler.fit(value)
    return scaler

#function creat train dataset in tensor
def creat_train_dataset(kind_processing):
    #create path file
    cwd = os.path.dirname(os.getcwd())

    if kind_processing == "normalize":
        dir_name = "norm_data_stft"
        
    else:
        dir_name = "stand_data_stft"
        
    path_dir = os.path.join(cwd,dir_name)    
    os.makedirs(path_dir,exist_ok=True)

    file = "train.pt"
    path_file = os.path.join(path_dir,file)

    if os.path.isfile(path_file) == False:
        
        #load train data
        data = DataPreprocessing("train")
        bearing_number = data.bearing_number["train"]
        sr = data.sr
        n_windows_train = data.n_windows(window_size,"train")
        
        #init X_train and y_train
        X_train = np.empty(shape = (n_windows_train,2,window_size))
        y_train = np.empty(shape = (n_windows_train,2))
        index_len = data.len_index(window_size=window_size,kind="train")
                                
        for index_train, bearing in enumerate (bearing_number):
            #create scaler
            scaler_bearing = scaler(bearing)

            #load parameter
            if kind_processing == "normalize":
                norm = data.norm(bearing)
            else:
                norm = data.stand(bearing)
            eol = data.eol(bearing)
            N = data.n_samples("train",bearing)
            n_windows_bearing = data.n_windows_bearing(window_size)["train"][bearing]
            
            #init X_train_bearing and label:
            X_train_bearing = np.empty(shape = (n_windows_bearing,2,window_size))
            y_train_bearing = np.empty(shape = (n_windows_bearing,2))

            for index,label in enumerate(range(0,eol+1-90,10)):
                #scale label
                label_array = np.array(label).reshape(-1,1)
                label_scaled = scaler_bearing.transform(label_array).item()

                #create step
                s_step = N - label * sr // 100  - window_size
                feature = norm[:,s_step:s_step+window_size].reshape(1,2,-1)

                #create couple feature, label, bearing
                X_train_bearing[index] = feature
                y_train_bearing[index] = [label_scaled,bearing]

            X_train[index_len[index_train]:index_len[index_train+1]] = X_train_bearing
            y_train[index_len[index_train]:index_len[index_train+1]] = y_train_bearing

        train_data = TensorDataset(torch.tensor(X_train),torch.tensor(y_train).float())
        torch.save(train_data,path_file)

#function creat test dataset in tensor
def creat_full_test_dataset(kind_processing):
    #create path file
    cwd = os.path.dirname(os.getcwd())

    if kind_processing == "normalize":
        dir_name = "norm_data_stft"
        
    else:
        dir_name = "stand_data_stft"
        
    path_dir = os.path.join(cwd,dir_name)    
    os.makedirs(path_dir,exist_ok=True)

    file = "full.pt"
    path_file = os.path.join(path_dir,file)

    if os.path.isfile(path_file) == False:
        
        #load train data
        data = DataPreprocessing("full")
        bearing_number = data.bearing_number["test"]
        sr = data.sr
        n_windows_train = data.n_windows(window_size,"full")
        
        #init X_train and y_train
        X_train = np.empty(shape = (n_windows_train,2,window_size))
        y_train = np.empty(shape = (n_windows_train,2))
        index_len = data.len_index(window_size=window_size,kind="full")
                                
        for index_test, bearing in enumerate (bearing_number):
            #create scaler
            scaler_bearing = scaler(bearing)

            #load parameter
            if kind_processing == "normalize":
                norm = data.norm(bearing)
            else:
                norm = data.stand(bearing)
            eol = data.eol(bearing)
            N = data.n_samples("full",bearing)
            n_windows_bearing = data.n_windows_bearing(window_size)["full"][bearing]
            
            #init X_train_bearing and label:
            X_test_bearing = np.empty(shape = (n_windows_bearing,2,window_size))
            y_test_bearing = np.empty(shape = (n_windows_bearing,2))

            for index,label in enumerate(range(0,eol+1-90,10)):
                #scale label
                label_array = np.array(label).reshape(-1,1)
                label_scaled = scaler_bearing.transform(label_array).item()

                #create step
                s_step = N - label * sr // 100  - window_size
                feature = norm[:,s_step:s_step+window_size].reshape(1,2,-1)
                X_test_bearing[index] = feature
                y_test_bearing[index] = [label_scaled,bearing]

            X_train[index_len[index_test]:index_len[index_test+1]] = X_test_bearing
            y_train[index_len[index_test]:index_len[index_test+1]] = y_test_bearing

        train_data = TensorDataset(torch.tensor(X_train),torch.tensor(y_train).float())
        torch.save(train_data,path_file)
        
#creat train and test data
creat_train_dataset("standardize")
creat_full_test_dataset("standardize")

end = default_timer()
print("hyperparameter tunning takes {} seconds".format(end - start))