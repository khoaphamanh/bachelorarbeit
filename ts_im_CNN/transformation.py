import os
import torch
import numpy as np
import librosa
import pywt

from torchvision import transforms
import torchvision
from torch.utils.data import TensorDataset, Dataset
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from pyts.image import RecurrencePlot,GramianAngularField,MarkovTransitionField
from tqdm import tqdm
from torch import nn
from torchinfo import summary
from pyts.approximation import PiecewiseAggregateApproximation as PAA

#class to load transition
class Transformation:
    def __init__(self, transform):
        
        #directory information
        self.femto  = os.path.dirname(os.getcwd())

        #images information
        self.chanel = 2
        self.w_size = 2560
        self.w_size_stft = 25600
        self.sr = 25600
        self.image_size = 224
        self.hop = 2560
        self.transform = transform
        
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
    def load_raw_data (self,kind_data, window_size):
        if window_size == 2560:
            data_dir = "data_2560"
        else:
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
        
        #clamp
        if clamp  == True:
            feature_train = np.clip(feature_train,-1,1)
            feature_val = np.clip(feature_val,-1,1)
             
        return TensorDataset(torch.from_numpy(feature_train).float(),label_train), TensorDataset(torch.from_numpy(feature_val).float(),label_val), scaler   
    
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

    def normalize_pixel (self,train_image,test_image):
        #normlize tensor from 0 to 1 on each chanel 
        feature_train, label_train = train_image.tensors
        feature_test, label_test = test_image.tensors
        
        #min and max on each chanel
        min_0 = torch.min(feature_train[:,0,:])
        min_1 = torch.min(feature_train[:,1,:])
        max_0 = torch.max(feature_train[:,0,:])
        max_1 = torch.max(feature_train[:,1,:])
    
        if max_0 == min_0 and max_1 == min_1:
            scaler_train = transforms.Normalize((0,0),(1,1))
            return train_image, test_image, scaler_train

        else:
            #create transform normalize
            scaler_train = transforms.Normalize((min_0,min_1),(max_0-min_0,max_1-min_1))
            
            return TensorDataset(scaler_train(feature_train),label_train) , TensorDataset(scaler_train(feature_test),label_test), scaler_train
    
    def load_model (self,drop,w_size):
        model = torchvision.models.efficientnet_b0()
        model.features[0][0] = nn.Conv2d(2,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier = nn.Identity()
        
        model = ConcatModel(model,drop,w_size)
        return model    
    
    def load_dataset (self, time_series:TensorDataset, image:TensorDataset):
        #split into feature and label
        feature_time_series, label = time_series.tensors
        feature_image, _ = image.tensors
        
        #create custom tensordataset
        my_dataset = CustomDataset(feature_time_series,feature_image,label)
        
        return my_dataset
    
    def merge_result (self,name_method:str):
        #merge all text file
        name_method = name_method.lower()
        dir_text = "result/{}".format(name_method)
        file_names = sorted([ file for file in os.listdir(dir_text) if file.endswith(".txt")])
        
        # Initialize an empty string to store merged content
        merged_content = ''

        # Loop through the list of file names and read and merge their contents
        for file_name in file_names:
            path_file_name = os.path.join(dir_text,file_name)
            with open(path_file_name, 'r') as file:
                content = file.read()
                merged_content += content + '\n'

        # Write the merged content to a new text file
        path_file_name = os.path.join(dir_text,'{}_final_result.txt'.format(name_method))
        with open(path_file_name, 'w') as merge_file:
            merge_file.write(merged_content)
            
        for file_name in file_names:
            os.remove(os.path.join(dir_text,file_name))    
            
    def stft(self, raw_data, w_stft = 2560, hop = 128):
        
        #load data
        len_data = len(raw_data)
        raw_data, label = raw_data.tensors
        
        #features transform tensor.
        feature_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))

        for index,feature in enumerate(raw_data):
        #for index,feature in enumerate(tqdm(raw_data)):    
            #create image:
            #_, _, feature_image = stft_scipy(x=feature,fs = self.sr,nperseg=w_stft,noverlap=w_stft-hop)
            feature_image = librosa.stft(feature.numpy(),n_fft=w_stft,hop_length=hop)
            feature_image = np.abs(feature_image)
            feature_image = librosa.power_to_db(feature_image)
            feature_image = torch.from_numpy(feature_image)
            
            #transform image
            feature_image = self.transform(feature_image)
            #indexing image and label
            feature_transform[index] = feature_image

        #create TensorDataset     
        data_transform = TensorDataset(feature_transform,label)
        return data_transform

    def rp (self,raw_data, threshold= None, percentage=50, dimension = 1, time_delay = 1):
        
        #load method
        RP = RecurrencePlot(dimension=dimension,time_delay=time_delay,threshold=threshold,percentage=percentage)
        paa = PAA(window_size=None,output_size=224)
        
        #load data
        raw_data, label = raw_data.tensors
        len_data = len(raw_data)

        #features transform tensor.
        feature_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))

        for index, feature in enumerate(raw_data):
        #for index,feature in enumerate(tqdm(raw_data,desc ="Processing {} Data")):    
            #create image:
            feature = paa.transform(feature)
            feature_image = RP.fit_transform(feature)
            feature_image = torch.from_numpy(feature_image)

            #indexing feature image and label
            feature_transform[index] = feature_image

        #create TensorDataset:
        data_transform = TensorDataset(feature_transform,label)
        return data_transform

    def lms(self,raw_data,w_stft=2560,hop=256,n_mels=256):
        
        #load data
        len_data = len(raw_data)
        raw_data, label = raw_data.tensors
        
        #features transform tensor.
        feature_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))
        
        for index,feature in enumerate(raw_data):
        #for index, feature in enumerate(tqdm(raw_data,desc ="Processing {} Data")):    
            #create image:
            feature_image = librosa.feature.melspectrogram(y=feature.numpy(),sr=self.sr,n_fft=w_stft,hop_length=hop,n_mels=n_mels)
            feature_image = librosa.power_to_db(feature_image)
            feature_image = torch.from_numpy(feature_image)
            
            #transform image
            feature_image = self.transform(feature_image)
            #indexing image and label
            feature_transform[index] = feature_image

        #create TensorDataset       
        data_transform = TensorDataset(feature_transform,label)
        return data_transform
    
    def gaf(self,raw_data,method:str="summation"):
       
        #load transition and scaler
        GAF = GramianAngularField(sample_range=None,method=method,image_size=224)
        
        #load data
        raw_data, label = raw_data.tensors
        len_data = len(raw_data)
        
        #features transform tensor.
        feature_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))

        for index,feature in enumerate(raw_data):
        #for index,feature in enumerate(tqdm(raw_data,desc ="Processing {} Data")):    
            #create image:
            feature_image = GAF.fit_transform(feature)
            feature_image = torch.from_numpy(feature_image)

            #indexing feature image and label
            feature_transform[index] = feature_image

        #create TensorDataset:
        data_transform = TensorDataset(feature_transform,label)
        
        return data_transform
    
    def mtf(self,raw_data,n_bins=5,strategy="quantile"):
        
        #load transition
        MTF = MarkovTransitionField(n_bins=n_bins,strategy=strategy)
        paa = PAA(window_size=None,output_size=224)
        
        #load data
        raw_data, label = raw_data.tensors
        len_data = len(raw_data)
        
        #features transform tensor.
        feature_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))
        
        for index,feature in enumerate(raw_data):
        #for index,feature in enumerate(tqdm(raw_data,desc ="Processing {} Data")):    
            #create image:
            feature = paa.transform(feature)
            feature_image = MTF.fit_transform(feature)
            feature_image = torch.from_numpy(feature_image)
                
            #indexing image and label
            feature_transform[index] = feature_image

        #create TensorDataset       
        data_transform = TensorDataset(feature_transform,label)
        return data_transform
    
    def cwt(self,raw_data,b:float = 1.5,fc:float = 1.0,scale_min:float = 2, scale_max:float = 20 ,n_scales:int=250):
        
        #load data
        raw_data, label = raw_data.tensors
        len_data = len(raw_data)
            
        #features transform tensor.
        feature_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))
        
        #load wavelet and scale
        wavelet = "cmor{}-{}".format(b,fc)
        scales = np.linspace(scale_min,scale_max,n_scales)
        
        for index,(feature_1,feature_2) in enumerate(raw_data):
        #for index,(feature_1,feature_2) in enumerate(tqdm(raw_data)):    
            #create image:
            feature_image_1, freq = pywt.cwt(data=feature_1.numpy(),scales=scales,wavelet=wavelet,sampling_period=1/self.sr)
            feature_image_2, freq = pywt.cwt(data=feature_2.numpy(),scales=scales,wavelet=wavelet,sampling_period=1/self.sr)
            
            feature_image = np.stack((feature_image_1,feature_image_2),axis=0)
            feature_image = np.abs(feature_image)
            #feature_image = librosa.power_to_db(feature_image)
            feature_image = torch.from_numpy(feature_image)
            
            #transform image
            feature_image = self.transform(feature_image)
            
            #indexing image and label
            feature_transform[index] = feature_image
            
        #create TensorDataset       
        data_transform = TensorDataset(feature_transform,label)
        return data_transform
    
class Conv1DBlock_1(nn.Module):
    def __init__(self,in_channels, out_channels,w_size):
        super().__init__()
        if w_size == 25600:
            kernel_size = 11
            stride = 5
            padding = 5
        else:
            kernel_size = 5
            stride = 2
            padding = 2
            
        self.conv1d = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size = kernel_size,
                               stride=stride,
                               padding =padding,
                               bias = True)
        
        self.batchnorm1d = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.conv1d(x)
        x = self.batchnorm1d(x)
        x = self.relu(x)
        return x

class Conv1DBlock_2(nn.Module):
    def __init__ (self,in_channels, out_channels):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size = 3, stride = 1,
                                  padding = 1, bias = True)
        self.batchnorm1d_1 = nn.BatchNorm1d(num_features=out_channels)
        self.relu_1 = nn.ReLU()
        
        self.conv1d_2 = nn.Conv1d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size = 3, stride = 1,
                                  padding = 1, bias = True)
        self.batchnorm1d_2  = nn.BatchNorm1d(num_features=out_channels)
        self.relu_2 = nn.ReLU()
        
        self.maxpood1d = nn.MaxPool1d(kernel_size = 4)
    
    def forward (self,x):
        x = self.conv1d_1(x)
        x = self.batchnorm1d_1(x)
        x = self.relu_1(x)
        
        x = self.conv1d_2(x)
        x = self.batchnorm1d_2(x)
        x = self.relu_2(x)
        
        x = self.maxpood1d(x)
        return x
 
class Conv2DBlock(nn.Module):
    def __init__ (self,in_channels, out_channels):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size = 3, stride = 1,
                                  padding = 1, bias = True)
        self.batchnorm2d_1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu_1 = nn.ReLU()
        
        self.conv2d_2 = nn.Conv2d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size = 3, stride = 1,
                                  padding = 1, bias = True)
        self.batchnorm2d_2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu_2 = nn.ReLU()
        
        self.maxpood2d = nn.MaxPool2d(kernel_size = (2,2))
        
    def forward(self,x):
        x = self.conv2d_1(x)
        x = self.batchnorm2d_1(x)
        x = self.relu_1(x)
        
        x = self.conv2d_2(x)
        x = self.batchnorm2d_2(x)
        x = self.relu_2(x)
        
        x = self.maxpood2d(x)
        return x
    
class Conv1dNeuralNetwork(nn.Module):
    def __init__ (self,w_size):
        super().__init__()
        self.conv1d_1 = Conv1DBlock_1(2,128,w_size)
        if w_size == 25600:
            self.conv1d_2 = nn.Sequential(Conv1DBlock_2(128,128),Conv1DBlock_2(128,128)) 
        else:
            self.conv1d_2 = Conv1DBlock_2(128,128)
        self.conv1d_3 = Conv1DBlock_2(128,128)

        
        self.conv2d_1 = Conv2DBlock(4,128)
        self.conv2d_2 = Conv2DBlock(128,128)
        self.conv2d_3 = Conv2DBlock(128,128)
        self.conv2d_4 = Conv2DBlock(128,128)
        
        self.flatten = nn.Flatten()
        
    def forward (self,x):
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        
        x = x.reshape(x.shape[0],-1,32,x.shape[-1]).transpose(2,3)      
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        
        x = self.flatten(x)

        return x

class ConcatModel (nn.Module):
    def __init__(self, efficient_net, drop,w_size):
        super(ConcatModel, self).__init__()
        self.model_3d = efficient_net
        self.model_1d = Conv1dNeuralNetwork(w_size=w_size)
        self.drop = nn.Dropout(p= drop)
        self.fc = nn.Linear(2 * 1280, 1)  # Output = 1

    def forward(self, x_1d, x_3d):
        x_1d = self.model_1d(x_1d)
        x_3d = self.model_3d(x_3d)
        x = torch.cat((x_1d, x_3d), dim=1)  # Concatenate the outputs
        x = self.drop(x)
        x = self.fc(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, time_series_data, image_data, labels):
        self.time_series_data = time_series_data
        self.image_data = image_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        time_series_input = self.time_series_data[index]
        image_input = self.image_data[index]
        label = self.labels[index]
        return time_series_input, image_input, label
             
if __name__ == "__main__":  

    from timeit import default_timer
    torch.manual_seed(1998)
    
    start = default_timer()
    
    transform_custom = transforms.Compose([
        transforms.Resize(size=(224,224),interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),
        
    ])
    
    tranformation = Transformation(transform=transform_custom)
    window_size = 25600
    model = tranformation.load_model(drop=0.2,w_size=window_size)
    
    train_data = tranformation.load_raw_data(kind_data="train",window_size=window_size)
    val_index = np.arange(0,2803).tolist()
    train_index = np.arange(2803,len(train_data)).tolist()
    
    train_data_norm, test_data_norm, _ =  tranformation.normalize_raw(train_data,train_index=train_index,val_index=val_index,min_max_scaler=False,clamp=False) #,val_data 
    lms = tranformation.lms(test_data_norm)
    
    my_dataset = tranformation.load_dataset(test_data_norm, lms)
    
    from torch.utils.data import Dataset, DataLoader
    batch_size = 10
    dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    
    for batch in dataloader:
        time_series_inputs, image_inputs, labels = batch
        # Process the inputs and labels as needed
        print("Time series inputs shape:", time_series_inputs.shape)
        print("Image inputs shape:", image_inputs.shape)
        print("Labels shape:", labels.shape)
        out = model(time_series_inputs,image_inputs)
        print("Outputs shape:", out.shape)
        print("---")
        
    end = default_timer()
    print(end-start)


