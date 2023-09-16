from helper import DataPreprocessing
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
from timeit import default_timer

start = default_timer()

#global variable
window_size = 2560
end_of_life = {11:28020,12:8700,21:9100,22:7960,31:5140,32:16360,13:23740,14:14270,15:24620,16:24470,17:22580,23:19540,24:7500,25:23100,26:7000,27:2290,33:4330}

#function to calculate the scaler: 0:dead, label = eol 1:healthy, label = begin
def scaler (bearing):
    eol = end_of_life[bearing]
    value = np.array([0,eol]).reshape(-1,1)
    min_max_scaler = MinMaxScaler(feature_range=(0,1))
    scaler = min_max_scaler.fit(value)
    return scaler

#function creat train dataset in tensor
def creat_dataset(kind_data:str,window_size:int):
    #create path file
    cwd = os.path.dirname(os.getcwd())
        
    path_dir = os.path.join(cwd,"data_{}".format(window_size))    
    os.makedirs(path_dir,exist_ok=True)

    file = "{}.pt".format(kind_data)
    path_file = os.path.join(path_dir,file)

    if os.path.isfile(path_file) == False:
        
        #load train data
        data = DataPreprocessing(kind_data)
        bearing_number = data.bearing_number[kind_data]
        sr = data.sr
        n_windows_data = data.n_windows(window_size,kind_data)
        
        #init X_train and y_train
        X_train = np.empty(shape = (n_windows_data,2,window_size))
        y_train = np.empty(shape = (n_windows_data,2))
        index_len = data.len_index(window_size=window_size,kind=kind_data)
                                
        for index_train, bearing in enumerate (bearing_number):
            #create scaler
            scaler_bearing = scaler(bearing)

            #load parameter
            raw = data.load_array(bearing)
            eol = data.eol(bearing)
            N = data.n_samples(kind_data,bearing)
            n_windows_bearing = data.n_windows_bearing(window_size)[kind_data][bearing]
            
            #init X_train_bearing and label:
            X_train_bearing = np.empty(shape = (n_windows_bearing,2,window_size))
            y_train_bearing = np.empty(shape = (n_windows_bearing,2))
            
            #range in loop:
            min_range = 0
            if window_size == 2560:
                max_range = eol + 1
            else:
                max_range = eol - 90 +1
                
            for index,label in enumerate(range(min_range,max_range,10)):
                #scale label
                label_array = np.array(label).reshape(-1,1)
                label_scaled = scaler_bearing.transform(label_array).item()

                #create step
                s_step = N - label * sr // 100  - window_size
                feature = raw[:,s_step:s_step+window_size].reshape(1,2,-1)

                #create couple feature, label, bearing
                X_train_bearing[index] = feature
                y_train_bearing[index] = [label_scaled,bearing]

            X_train[index_len[index_train]:index_len[index_train+1]] = X_train_bearing
            y_train[index_len[index_train]:index_len[index_train+1]] = y_train_bearing

        train_data = TensorDataset(torch.tensor(X_train),torch.tensor(y_train).float())
        torch.save(train_data,path_file)


#creat train and test data
creat_dataset("train",2560)
creat_dataset("full",2560)

creat_dataset("train",25600)
creat_dataset("full",25600)

end = default_timer()
print("hyperparameter tunning takes {} seconds".format(end - start))