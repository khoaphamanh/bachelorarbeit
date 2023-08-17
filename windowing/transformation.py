import os
import torch
import numpy as np
import librosa
import pywt

from torchvision import transforms,models
from torch.utils.data import TensorDataset
from scipy.signal import stft as stft_scipy
from sklearn.preprocessing import MinMaxScaler
from pyts.image import RecurrencePlot,GramianAngularField,MarkovTransitionField
from tqdm import tqdm

#class to load transition
class Transformation:
    def __init__(self,transform):
        
        #directory information
        self.cwd  = os.path.dirname(os.getcwd())
        self.norm = "normalize"
        self.stand = "standardize"
        self.stand_stft = "standardize_stft"
        self.norm_data = "norm_data"
        self.stand_data = "stand_data"
        self.stand_data_stft = "stand_data_stft"
        self.norm_data_dir = os.path.join(os.path.dirname(os.getcwd()),self.norm_data)
        self.stand_data_dir = os.path.join(os.path.dirname(os.getcwd()),self.stand_data)

        #images information
        self.chanel = 2
        self.w_size = 2560
        self.w_size_stft = 25600
        self.sr = 25600
        self.image_size = 224
        self.transform = transform
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
    
    def bearing_cv_index(self,window_size):
        len_index_train = self.len_index(window_size=window_size,kind="train")
        indices = np.arange(len_index_train[0],len_index_train[-1]).tolist()
        val_indices = [np.arange(len_index_train[i],len_index_train[i+1]).tolist() for i in range(len(len_index_train)-1)]
        #print("val_index:", val_index)
        #print("val_index:", len(val_index))
        for val_index in val_indices:
            train_indices = [i for i in indices if i not in val_index]
            print("train_indices:", (train_indices))
            print("val_index", len(val_index))        
    
    #function to load data
    def load_data (self,kind_data, kind_processing):
    
        if kind_processing == "normalize":
            path_process = os.path.join(self.cwd,self.norm_data)

        elif kind_processing == "standardize":
            path_process = os.path.join(self.cwd,self.stand_data)

        elif kind_processing == "standardize_stft":
            path_process = os.path.join(self.cwd,self.stand_data_stft)

        if kind_data == "train":
            file = "train.pt"
            path_file = os.path.join(path_process,file)
            train_data = torch.load(path_file)
            return train_data
        
        elif kind_data == "full":
            file = "full.pt"
            path_file = os.path.join(path_process,file)
            test_data = torch.load(path_file)
            return test_data
        
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

    def normalize (self,images):
        #normlize tensor from 0 to 1 on each chanel 
        #min and max on each chanel
        min_0 = torch.min(images[:,0,:])
        min_1 = torch.min(images[:,1,:])
        max_0 = torch.max(images[:,0,:])
        max_1 = torch.max(images[:,1,:])
        
        #create transform normalize
        norm = transforms.Normalize((min_0,min_1),(max_0-min_0,max_1-min_1))
        return norm(images)
    
    def stft(self,kind_data, w_stft = 2560, hop = 128):
        
        #load data
        data = self.load_data(kind_data=kind_data,kind_processing=self.stand_stft)
        len_data = len(data)
        raw_data, label = data.tensors
        
        #features transform tensor.
        fearture_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))

        #for index,feature in enumerate(raw_data):
        for index,feature in enumerate(tqdm(raw_data)):    
            #create image:
            #_, _, feature_image = stft_scipy(x=feature,fs = self.sr,nperseg=w_stft,noverlap=w_stft-hop)
            feature_image = librosa.stft(feature.numpy(),n_fft=w_stft,hop_length=hop)
            feature_image = np.abs(feature_image)
            feature_image = librosa.power_to_db(feature_image)
            feature_image = torch.from_numpy(feature_image)
            
            #transform image
            feature_image = self.transform(feature_image)
            #indexing image and label
            fearture_transform[index] = feature_image

        #create TensorDataset
        fearture_transform = self.normalize(fearture_transform)       
        data_transform = TensorDataset(fearture_transform,label)
        return data_transform

    def rp (self,kind_data, threshold, percentage,dimension = 1, time_delay = 1, bearing=None, padding = False):
        
        #load method
        RP = RecurrencePlot(dimension=dimension,time_delay=time_delay,threshold=threshold,percentage=percentage)

        #load data
        data = self.load_data(kind_data=kind_data,kind_processing=self.norm,bearing=bearing)
        len_data = len(data)

        #check padding 
        if padding == True:
            fearture_transform = torch.empty(size=(len_data,3,self.image_size,self.image_size))
        elif padding == False:
            fearture_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))

        fearture_label = torch.empty(size=(len_data,2))

        for index,(feature, label) in enumerate(data):
        #for index,(feature, label) in enumerate(tqdm(data,desc ="Processing {} Data".format(kind_data))):    
            #create image:
            fearture_image = RP.fit_transform(feature)

            #check padding
            if padding == True:
                fearture_image = self.padding_3_chanel(torch.from_numpy(fearture_image))
            else:
                fearture_image = torch.from_numpy(fearture_image)

            #transform image
            fearture_image = self.transform(fearture_image)

            #indexing feature image and label
            fearture_transform[index] = fearture_image
            fearture_label[index] = label

        #create TensorDataset:
        data_transform = TensorDataset(fearture_transform,fearture_label)
        return data_transform

    def ms(self,kind_data,w_stft,hop,n_mels,bearing=None,padding =False):
        
        #load data
        data = self.load_data(kind_data=kind_data,kind_processing=self.stand_stft,bearing=bearing)
        len_data = len(data)
        
        #check padding
        if padding == True:
            fearture_transform = torch.empty(size=(len_data,3,self.image_size,self.image_size))
        elif padding == False:
            fearture_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))

        fearture_label = torch.empty(size=(len_data,2))
        
        for index,(feature, label) in enumerate(data):
        #for index,(feature, label) in enumerate(tqdm(data,desc ="Processing {} Data".format(kind_data))):    
            #create image:
            fearture_image = librosa.feature.melspectrogram(y=feature.numpy(),sr=self.sr,n_fft=w_stft,hop_length=hop,n_mels=n_mels)
            fearture_image = librosa.power_to_db(fearture_image)
            
            #check padding:
            if padding == True:
                fearture_image = self.padding_3_chanel(torch.from_numpy(fearture_image))
            else:
                fearture_image = torch.from_numpy(fearture_image)
            #transform image
            fearture_image = self.transform(fearture_image)
            #indexing image and label
            fearture_transform[index] = fearture_image
            fearture_label[index] = label

        #create TensorDataset       
        data_transform = TensorDataset(fearture_transform,fearture_label)
        return data_transform
    
    def gaf(self,kind_data:str,method:str="summation",bearing=None, padding = False):
       
       #load transition and scaler
        GAF = GramianAngularField(sample_range=None,method=method)
        
        #load data
        data = self.load_data(kind_data,self.norm, bearing)
        len_data = len(data)
        
        #check padding 
        if padding == True:
            fearture_transform = torch.empty(size=(len_data,3,self.image_size,self.image_size))
        elif padding == False:
            fearture_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))

        fearture_label = torch.empty(size=(len_data,2))

        for index,(feature, label) in enumerate(data):
        #for index,(feature, label) in enumerate(tqdm(data,desc ="Processing {} Data".format(kind_data))):    
            #create image:
            fearture_image = GAF.fit_transform(feature)

            #check padding
            if padding == True:
                fearture_image = self.padding_3_chanel(torch.from_numpy(fearture_image))
            else:
                fearture_image = torch.from_numpy(fearture_image)

            #transform image
            fearture_image = self.transform(fearture_image)

            #indexing feature image and label
            fearture_transform[index] = fearture_image
            fearture_label[index] = label

        #create TensorDataset:
        data_transform = TensorDataset(fearture_transform,fearture_label)
        return data_transform
    
    def mtf(self,kind_data,n_bins=5,strategy="quantile",bearing=None,padding=False):
        
        #load transition
        MTF = MarkovTransitionField(n_bins=n_bins,strategy=strategy)
        
        #load data
        data = self.load_data(kind_data=kind_data,kind_processing=self.norm,bearing=bearing)
        len_data = len(data)
        
        #check padding
        if padding == True:
            fearture_transform = torch.empty(size=(len_data,3,self.image_size,self.image_size))
        elif padding == False:
            fearture_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))

        fearture_label = torch.empty(size=(len_data,2))
        
        for index,(feature, label) in enumerate(data):
        #for index,(feature, label) in enumerate(tqdm(data,desc ="Processing {} Data".format(kind_data))):    
            #create image:
            fearture_image = MTF.fit_transform(feature)
            fearture_image = librosa.power_to_db(fearture_image)
            
            #check padding:
            if padding == True:
                fearture_image = self.padding_3_chanel(torch.from_numpy(fearture_image))
            else:
                fearture_image = torch.from_numpy(fearture_image)
            #transform image
            fearture_image = self.transform(fearture_image)
            #indexing image and label
            fearture_transform[index] = fearture_image
            fearture_label[index] = label

        #create TensorDataset       
        data_transform = TensorDataset(fearture_transform,fearture_label)
        return data_transform
    
    def cwt(self,kind_data,b:float,fc:float,scale_min:float, scale_max:float,n_scales:int,bearing=None,padding =False):
        
        #load data
        data = self.load_data(kind_data=kind_data,kind_processing=self.stand,bearing=bearing)
        len_data = len(data)
        
        #check padding
        if padding == True:
            fearture_transform = torch.empty(size=(len_data,3,self.image_size,self.image_size))
        elif padding == False:
            fearture_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))
            
        fearture_label = torch.empty(size=(len_data,2))    
        
        #load wavelet and scale
        wavelet = "cmor{}-{}".format(b,fc)
        scales = np.linspace(scale_min,scale_max,n_scales)
        
        for index,((feature_1,feature_2), label) in enumerate(data):
        #for index,((feature_1,feature_2), label) in enumerate(tqdm(data,desc ="Processing {} Data".format(kind_data))):    
            #create image:
            fearture_image_1, freq = pywt.cwt(data=feature_1.numpy(),scales=scales,wavelet=wavelet,sampling_period=1/self.sr)
            fearture_image_2, freq = pywt.cwt(data=feature_2.numpy(),scales=scales,wavelet=wavelet,sampling_period=1/self.sr)
            
            fearture_image = np.stack((fearture_image_1,fearture_image_2),axis=0)
            fearture_image = np.abs(fearture_image)

            #check padding:
            if padding == True:
                fearture_image = self.padding_3_chanel(torch.from_numpy(fearture_image))
            else:
                fearture_image = torch.from_numpy(fearture_image)
                
            #transform image
            fearture_image = self.transform(fearture_image)
            
            #indexing image and label
            fearture_transform[index] = fearture_image
            fearture_label[index] = label
            
        #create TensorDataset       
        data_transform = TensorDataset(fearture_transform,fearture_label)
        return data_transform
                          
if __name__ == "__main__":  

    from timeit import default_timer
    start = default_timer()

    #pretrained weight from pretrained model
    weights = models.EfficientNet_B0_Weights.DEFAULT

    transform_efb0 = weights.transforms(antialias=True)

    transform_custom = transforms.Compose([
        transforms.Resize(size=(224,224),interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),
        
    ])
    
    tranformation = Transformation(transform=transform_custom)

    lc_index = tranformation.lc_index(window_size=2560)
    print("lc_index:", lc_index)
    """bearing_cv_index = tranformation.bearing_cv_index(window_size=2560)
    print("bearing_cv_index:", bearing_cv_index)"""
      
    """norm_data_dir = tranformation.norm_data_dir
    print("norm_data_dir:", norm_data_dir)
    stand_data_dir = tranformation.stand_data_dir
    print("stand_data_dir:", stand_data_dir)"""

    """data = tranformation.load_data("train","normalize") 
    print(data[0][0].shape)"""
    
    """#scaler = tranformation.scaler
    inverse_transform = tranformation.inverse_transform_label
    bearing = 11
    y_scaled = torch.tensor(0.1)
    y = inverse_transform(bearing,y_scaled)
    print(y)
    
    #scaler = tranformation.scaler
    inverse_transform = tranformation.inverse_transform_label
    bearing = 12
    y_scaled = torch.tensor(0.1)
    y = inverse_transform(bearing,y_scaled)
    print(y)"""

    """STFT = tranformation.stft("train",w_stft=2560,hop=256)  
    print(STFT)
    print(len(STFT))
    print(STFT[2793])
    print(STFT[2793][0].shape)
    print(STFT[2793][1])
    
    import matplotlib.pyplot as plt
    image = STFT[2793][0][0]
    plt.imshow(image,aspect="auto", interpolation="none", origin='lower')
    plt.colorbar()
    #plt.show()"""
    
    """import matplotlib.pyplot as plt
    image = STFT[0][0][0]
    plt.imshow(image,aspect="auto", interpolation="none")
    plt.show()"""
    
    
    """GASF = tranformation.gaf("train",padding=False)  
    print(len(GASF))
    print(GASF[2802])
    print(GASF[2802][0].shape)
    print(GASF[2802][1])"""
    
    """MTF = tranformation.mtf("train")  
    print(MTF)
    print(MTF[2802])
    print(MTF[2801][0].shape)
    print(MTF[2801][1])""" 

    #a = np.array ([1,0.5,0.2,0]).reshape(-1,1)
    #a_real = scaler_label(a)
    #print(a_real)

    """RP = tranformation.rp("train",threshold="point",percentage=31,dimension=1,time_delay=1,bearing=None,padding=False)  
    print(RP)
    print(RP[2802])
    print(RP[2802][0].shape)
    print(RP[2802][1])"""
    
    """MS = tranformation.ms("train",w_stft=2560,hop=256,n_mels=100,padding=False)  
    print(MS)
    print(MS[2802])
    print(MS[2802][0].shape)
    print(MS[2802][1])"""

    """CWT = tranformation.cwt(kind_data="train",b=1.5,fc = 1,scale_min=1,scale_max=20,n_scales=50)  
    print(CWT)
    print(CWT[2802])
    print(CWT[2802][0].shape)
    print(CWT[2802][1])"""
    
    end = default_timer()
    print(end-start)
