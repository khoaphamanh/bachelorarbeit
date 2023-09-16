import os
import torch
import numpy as np

from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from pre_trained import Wavegram_Logmel_Cnn14
from torch import nn
from torchinfo import summary

#class to load transition
class Transformation:
    def __init__(self):
        
        #directory information
        self.femto  = os.path.dirname(os.getcwd())

        #images information
        self.chanel = 2
        self.w_size = 2560
        self.w_size_stft = 25600
        self.sr = 25600
        self.image_size = 224
        self.hop = 2560
        
        #bearing information
        self.bearing_train = [11,12,21,22,31,32]
        self.eol = {11:28020,12:8700,21:9100,22:7960,31:5140,32:16360,13:23740,14:14270,15:24620,16:24470,17:22580,23:19540,24:7500,25:23100,26:7000,27:2290,33:4330}

        self.bearing_test = [13,14,15,16,17,23,24,25,26,27,33]
        self.last_cycle =  {13: 18010, 14: 11380, 15: 23010, 16: 23010, 17: 15010, 23: 12010, 24: 6110, 25: 20010, 26: 5710, 27: 1710, 33: 3510}
        self.rul = [5730,2890,1610,1460,7570,7530,1390,3090,1290,580,820]
        
        self.bearing_number = {"train":[11,12,21,22,31,32], "test":[13,14,15,16,17,23,24,25,26,27,33]}
        
        #scaler     
        self.scaler = {k: self.scaler_label(k) for k in self.bearing_train+self.bearing_test}

    def n_samples(self,kind:str, bearing = None):
        #num samples of train and test data
        num_samples = {"train":{11: 7175680, 12: 2229760, 21: 2332160, 22: 2040320, 31: 1318400, 32: 4190720}, 
                       "test":{13: 4613120, 14: 2915840, 15: 5893120, 16: 5893120, 17: 3845120, 23: 3077120,24: 1566720, 25: 5125120, 26: 1464320, 27: 440320, 33: 901120},
                       "full" : {13: 6080000, 14: 3655680, 15: 6305280, 16: 6266880, 17: 5783040, 23: 5004800, 24: 1922560, 25: 5916160, 26: 1794560, 27: 588800, 33: 1111040}}
        return num_samples[kind][bearing]
    
    def n_windows (self,window_size,kind : str):
        #calculate the number of windows in each bearing, train and test data an       
        if kind == "train":
            sum_train = 0
            for bear in self.bearing_number[kind]:
                sum_train = sum_train + (self.n_samples(kind,bear)-window_size)// self.hop + 1
            return sum_train
       
        elif kind == "test":
            sum_test = 0
            for bear in self.bearing_number[kind]:
                sum_test = sum_test + (self.n_samples(kind,bear)-window_size)// self.hop + 1 
            return sum_test
        
        elif kind == "full":
            sum_test = 0
            for bear in self.bearing_number["test"]:
                sum_test = sum_test + (self.n_samples(kind,bear)-window_size)// self.hop + 1 
            return sum_test

    def n_windows_bearing (self,window_size):
        #num windows of each bearing in train/ test/ full data
        bearing_number = self.bearing_number
        bearing_number["full"] = self.bearing_number["test"]
        bearing_windows = {"train":{},"test":{},"full":{}}
        for kind in bearing_windows.keys():
            for bear in bearing_number[kind]:
                bearing_windows[kind][bear] = (self.n_samples(kind,bear)-window_size)// self.hop + 1 
        return bearing_windows
        
    def len_index (self,window_size,kind:str):
        #len index of each bearing in train/ test/ full data
        bearing_windows = self.n_windows_bearing(window_size=window_size)[kind]
        len_index_ = np.cumsum([bearing_windows[key] for key in bearing_windows.keys()])
        len_index_ = np.insert(len_index_,0,0)
        return len_index_            
    
    def test_full_index(self,window_size):
        #index test data from full test data
        n_windows = self.n_windows_bearing(window_size=window_size)["test"]
        n_windows = [n_windows[key] for key in n_windows.keys()]
        full = self.len_index(window_size=window_size,kind="full")

        index_test = []
        for index, n_wins in enumerate(n_windows):
            tmp = [i for i in range(full[index+1]-n_wins,full[index+1])]
            index_test =  index_test + tmp
            
        return index_test
    
    def lc_index (self, window_size):
        #index of the last cycle in test data (cut point of full test)       
        n_windows = self.n_windows_bearing(window_size=window_size)["test"]
        n_windows = [n_windows[key] for key in n_windows.keys()]
        full = self.len_index(window_size=window_size,kind="full")
        
        lc_idx = {}
        for index, n_wins in enumerate(n_windows):
            lc_idx.setdefault(self.bearing_number["test"][index],full[index+1]-n_wins)
        return lc_idx
    
    def lc_window_index (self, window_size):
        #index of the last cycle in bearing already windowing       
        n_windows = self.n_windows_bearing(window_size=window_size)["test"]
        n_windows = {key:value-1 for key,value in n_windows.items()}
        
        return n_windows
    
    #function to load data
    def load_raw_data (self,kind_data,):
        data_dir = "data_25600"
        path_data = os.path.join(self.femto,data_dir)
        if kind_data == "train":
            file = "train.pt"
        else :
            file = "full.pt"
        path_file = os.path.join(path_data,file)
        return torch.load(path_file)
    
    def normalize_raw (self,train_val_raw_data:TensorDataset, test_raw_data:TensorDataset = None,train_index=None, val_index=None, min_max_scaler = True,clamp = False):
        #funtion to split features and labels from tensor
        if train_index is not None and val_index is not None:
            feature_raw, label_raw = train_val_raw_data.tensors
            feature_train, feature_val = feature_raw[train_index], feature_raw[val_index]
            label_train, label_val = label_raw[train_index], label_raw[val_index]
        else:
            feature_train, label_train = train_val_raw_data.tensors
            feature_val, label_val = test_raw_data.tensors
            
        #reshape
        chanel = 2
        feature_train = feature_train.transpose(1,2).reshape(-1,chanel)
        feature_val = feature_val.transpose(1,2).reshape(-1,chanel)
        
        #normalize
        if min_max_scaler == True:
            scaler = MinMaxScaler(feature_range=(-1,1))
        else:
            scaler =  StandardScaler()
            
        #fit the scaler
        scaler.fit(feature_train)
        
        #transform the data
        feature_train = scaler.transform(feature_train).reshape(len(label_train),-1,chanel)
        feature_train = np.transpose(feature_train,(0,2,1))
        feature_val = scaler.transform(feature_val).reshape(len(label_val),-1,chanel) 
        feature_val = np.transpose(feature_val,(0,2,1))
        
        #padding 
        #feature_train = np.pad(feature_train,pad_width=((0,0),(0,0),(3200,3200)),mode="constant",constant_values = 0)
        #feature_val = np.pad(feature_val,pad_width=((0,0),(0,0),(3200,3200)),mode="constant",constant_values = 0)
        
        #clamp
        if clamp  == True:
            feature_val = np.clip(feature_val,-1,1)
            
        return TensorDataset(torch.from_numpy(feature_train),label_train), TensorDataset(torch.from_numpy(feature_val),label_val), scaler            
    
    #function to scale the label to range
    def scaler_label(self,bearing):
        eol = self.eol[bearing]
        value = np.array([0,eol]).reshape(-1,1)
        min_max_scaler = MinMaxScaler(feature_range=(0,1))
        scaler = min_max_scaler.fit(value)
        return scaler
    
    #function to inverse transform label for plot in dict {bearing:[[rul_pred],[rul_true]]}
    def inverse_transform_label (self,bearing,label):
        #inverse transform
        scaler = self.scaler[int(bearing)]
        if torch.is_tensor(label):
            label = label.reshape(-1,1).clone().detach()
            label = scaler.inverse_transform(label).ravel()
            return torch.tensor(label)
        else:
            label = label.reshape(-1,1)
            label = scaler.inverse_transform(label).ravel()
            return label.item()     
        
    def load_model (self,drop,device):
        #state_dict = torch.load("Wavegram_Logmel_Cnn14_mAP=0.439.pth", map_location=torch.device(device))['model']
        model_1 = Wavegram_Logmel_Cnn14(sample_rate=self.sr,window_size=1024,hop_size=320,mel_bins=64,fmin=0,fmax=self.sr//2)
        #model_1.load_state_dict(state_dict,strict=False)
        model_2 = Wavegram_Logmel_Cnn14(sample_rate=self.sr,window_size=1024,hop_size=320,mel_bins=64,fmin=0,fmax=self.sr//2)
        #model_2.load_state_dict(state_dict,strict=False)
        concat_model = ConcatModel(model_1=model_1,model_2=model_2,drop=drop)
        """for name,params in concat_model.named_parameters():
            if not name.startswith("fc"):
                params.requires_grad = False"""
        return concat_model

    def merge_result (self,name_method):
        #merge all text file
        dir_text = "result/{}".format(name_method)
        file_names = sorted([ file for file in os.listdir(dir_text) if file.endswith(".txt") and not file.startswith("final")])
        
        # Initialize an empty string to store merged content
        merged_content = ''

        # Loop through the list of file names and read and merge their contents
        for file_name in file_names:
            path_file_name = os.path.join(dir_text,file_name)
            with open(path_file_name, 'r') as file:
                content = file.read()
                merged_content += content + '\n'

        # Write the merged content to a new text file
        path_file_name = os.path.join(dir_text,'final_result.txt')
        with open(path_file_name, 'w') as merge_file:
            merge_file.write(merged_content)
            
        
class ConcatModel(nn.Module):
    def __init__(self,model_1,model_2,drop):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.fc1 = nn.Linear(in_features=4096,out_features=512)
        self.drop_out = nn.Dropout(p=drop)
        self.fc2 = nn.Linear(in_features=512,out_features=1)
    def forward(self,x_hor, x_ver):
        out = torch.cat((self.model_1(x_hor),self.model_2(x_ver)),dim=1)
        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        return out
            
if __name__ == "__main__":  

    from timeit import default_timer
    torch.manual_seed(1998)
    start = default_timer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)    
    
    transformation =Transformation()
    """train_data = transformation.load_raw_data(kind_data="train")
    test_data = transformation.load_raw_data(kind_data="test")
    
    train_data, test_data =  transformation.normalize_raw(train_data,test_data,min_max_scaler=False) #,val_data 
    print("train_data:", len(train_data))
    print("test_data:", len(test_data))"""
    
    model = transformation.load_model(drop=0.2,device=device)
    #print("model:", model)
    summary(model ,col_names=[ "num_params", "trainable"])
    x_ver = torch.randn(10,25600)
    x_hor = torch.randn(10,25600)
    out = model(x_ver,x_hor)
    print("out:", out.shape)
    end = default_timer()
    print(end-start)
