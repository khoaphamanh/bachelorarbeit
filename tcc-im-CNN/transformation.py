import os
import torch
import numpy as np
import librosa
import pywt

from torchvision import transforms,models
from torch.utils.data import TensorDataset, ConcatDataset
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from pyts.image import RecurrencePlot,GramianAngularField,MarkovTransitionField
from tqdm import tqdm
from pyts.approximation import PiecewiseAggregateApproximation as PAA

#class to load transition
class Transformation:
    def __init__(self,transform):
        
        #directory information
        self.femto  = os.path.dirname(os.getcwd())

        #images information
        self.chanel = 2
        self.w_size = 2560
        self.w_size_stft = 25600
        self.sr = 25600
        self.image_size = 224
        self.transform = transform
        self.hop = 25600
        
        #bearing information
        self.bearing_train = [11,12,21,22,31,32]
        self.eol = {11:28020,12:8700,21:9100,22:7960,31:5140,32:16360,13:23740,14:14270,15:24620,16:24470,17:22580,23:19540,24:7500,25:23100,26:7000,27:2290,33:4330}

        self.bearing_test = [13,14,15,16,17,23,24,25,26,27,33]
        self.last_cycle =  {13: 18010, 14: 11380, 15: 23010, 16: 23010, 17: 15010, 23: 12010, 24: 6110, 25: 20010, 26: 5710, 27: 1710, 33: 3510}
        self.rul = [5730,2890,1610,1460,7570,7530,1390,3090,1290,580,820]
        
        self.bearing_number = {"train":[11,12,21,22,31,32], "test":[13,14,15,16,17,23,24,25,26,27,33], "full":[13,14,15,16,17,23,24,25,26,27,33]}
        
        #scaler     
        self.scaler = {k: self.scaler_label(k) for k in self.bearing_train+self.bearing_test}

    def n_samples(self,kind:str, bearing = None):
        #num samples of train and test data
        num_samples = {"train":{11: 7175680, 12: 2229760, 21: 2332160, 22: 2040320, 31: 1318400, 32: 4190720}, 
                       "test":{13: 4613120, 14: 2915840, 15: 5893120, 16: 5893120, 17: 3845120, 23: 3077120,24: 1566720, 25: 5125120, 26: 1464320, 27: 440320, 33: 901120},
                       "full" : {13: 6080000, 14: 3655680, 15: 6305280, 16: 6266880, 17: 5783040, 23: 5004800, 24: 1922560, 25: 5916160, 26: 1794560, 27: 588800, 33: 1111040}}
        return num_samples[kind][bearing]
    
    def n_windows (self,window_size,kind : str):
        #calculate the number of total windows in train/ test/ full data   
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
    
    def bearing_indices (self,window_size, kind):
        #funtion to get the index of each bearing in dataset in a dict with {bearing : index of bearings in dataset}
        dict_indices = {}
        len_index = self.len_index(window_size=window_size,kind=kind)
        for index,bearing in enumerate(self.bearing_number[kind]):
            indices = np.arange(len_index[index],len_index[index+1])
            dict_indices.setdefault(bearing,indices)
        
        return dict_indices
                    
    def load_raw_data (self,kind_data, window_size):
        #function to load data in dict with {beaing:dataset} 
        data_dict = {}
        
        #get the path 
        data_dir = "data_tcc"
        path_data = os.path.join(self.femto,data_dir)
        if kind_data == "train":
            file = "train.pt"
        else :
            file = "full.pt"
            
        path_file = os.path.join(path_data,file)
        
        #load dataset as TensorDatasets
        dataset = torch.load(path_file)
        feature, label = dataset.tensors

        #save dataset to dict
        bearing_index = self.bearing_indices(window_size=window_size, kind=kind_data)
        for bearing in self.bearing_number[kind_data]:
            feature_sub, label_sub = torch.flip(feature[bearing_index[bearing]],dims = [0]), torch.flip(label[bearing_index[bearing]],dims = [0])
            data_dict[bearing] = TensorDataset(feature_sub,label_sub)
            
        return data_dict
    
    def normalize_raw (self,train_val_raw_data:dict, test_raw_data:dict = None,train_bearing:list=None, val_bearing:int=None,min_max_scaler = True,clamp = False):
        #funtion to split features and labels from tensor, use for cv if bearing train and bearing val is available and test.
        
        #create train and val data in cv
        if train_bearing is not None and val_bearing is not None:
            feature_train = []
            for bearing in train_bearing:
                tensor_value = train_val_raw_data[bearing]
                feature_train.append(tensor_value.tensors[0])
            feature_train = torch.concat(feature_train,dim =0)
            
            feature_val =  []
            tensor_value = train_val_raw_data[val_bearing]
            feature_val.append(tensor_value.tensors[0])
            feature_val = torch.concat(feature_val,dim =0)

        #create train and test data in testing
        else:
            feature_train = []
            for tensor_value in train_val_raw_data.values():
                feature_train.append(tensor_value.tensors[0])
                label_train.append(tensor_value.tensors[1])
            feature_train = torch.concat(feature_train,dim =0)
            
            feature_val = []
            for tensor_value in test_raw_data.values():
                feature_val.append(tensor_value.tensors[0])
            feature_val = torch.concat(feature_val,dim =0)
            
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
        
        #from feature, label back to dict with form {bearing: tensordataset} in cv 
        if train_bearing is not None and val_bearing is not None:
            train_data = {}
            val_data = {}
            for bearing in train_bearing:
                tensor_value = train_val_raw_data[bearing]
                feature_train, label_train = tensor_value.tensors[0],tensor_value.tensors[1]
                feature_train = feature_train.transpose(1,2).reshape(-1,chanel)
                feature_train = scaler.transform(feature_train).reshape(len(label_train),-1,chanel)
                feature_train = torch.from_numpy(np.transpose(feature_train,(0,2,1)))
                if clamp  == True:
                    feature_train = torch.clamp(feature_train,-1,1)
                tensor_value = TensorDataset(feature_train,label_train)
                train_data.setdefault(bearing,tensor_value)
            
            tensor_value = train_val_raw_data[val_bearing]
            feature_val, label_val = tensor_value.tensors[0],tensor_value.tensors[1]
            feature_val = feature_val.transpose(1,2).reshape(-1,chanel)
            feature_val = scaler.transform(feature_val).reshape(len(label_val),-1,chanel)
            feature_val = torch.from_numpy(np.transpose(feature_val,(0,2,1)))
            if clamp  == True:
                feature_val = torch.clamp(feature_val,-1,1)
            tensor_value = TensorDataset(feature_val,label_val)
            val_data.setdefault(val_bearing,tensor_value)
        
        #from feature, label back to dict with form {bearing: tensordataset} in testing 
        else:
            train_data = {}
            val_data = {}
            for bearing in self.bearing_train:
                tensor_value = train_val_raw_data[bearing]
                feature_train, label_train = tensor_value.tensors[0],tensor_value.tensors[1]
                feature_train = feature_train.transpose(1,2).reshape(-1,chanel)
                feature_train = scaler.transform(feature_train).reshape(len(label_train),-1,chanel)
                feature_train = torch.from_numpy(np.transpose(feature_train,(0,2,1)))
                if clamp  == True:
                    feature_train = torch.clamp(feature_train,-1,1)
                tensor_value = TensorDataset(feature_train,label_train)
                train_data.setdefault(bearing,tensor_value)
            
            for bearing in self.bearing_test:
                tensor_value = test_raw_data[bearing]
                feature_val, label_val = tensor_value.tensors[0],tensor_value.tensors[1]
                feature_val = feature_val.transpose(1,2).reshape(-1,chanel)
                feature_val = scaler.transform(feature_val).reshape(len(label_val),-1,chanel)
                feature_val = torch.from_numpy(np.transpose(feature_val,(0,2,1)))
                if clamp  == True:
                    feature_val = torch.clamp(feature_val,-1,1)
                tensor_value = TensorDataset(feature_val,label_val)
                val_data.setdefault(bearing,tensor_value)
                
        return train_data, val_data, scaler            
    
    #function to shuffle dict dataset:
    def shuffle_dict_dataset (self,dict_dataset:dict):
        items = list(dict_dataset.items())
        np.random.shuffle(items)
        shuffle_dict =  dict(items)
        return shuffle_dict

    #function to create pairs of bearing
    def create_pairs (self,train_data:dict, test_data:dict=None,shuffle = True):
        pairs = []
        if shuffle:
            train_data = self.shuffle_dict_dataset(train_data)
        else:
            train_data = {key: train_data[key] for key in sorted(train_data.keys())}
        
        list_keys_train = list(train_data.keys())
        if test_data is None:
            for i in range(len(list_keys_train)):
                for j in range(i+1,len(list_keys_train)):
                    pairs.append([list_keys_train[i],list_keys_train[j]])
            return pairs
        
        else:
            if shuffle:
                test_data = self.shuffle_dict_dataset(test_data)
            else:
                test_data = {key: test_data[key] for key in sorted(test_data.keys())}
            list_keys_test = list(test_data.keys())
            for i in range(len(list_keys_test)):
                for j in range(len(list_keys_train)):
                    pairs.append([list_keys_test[i],list_keys_train[j]])
                    
            return pairs
    
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
        
    def normalize_pixel (self,train_image:dict,test_image:dict):
        #normlize tensor from 0 to 1 on each chanel 
        feature_train = []
        for key in train_image.keys():
            tensor_value = train_image[key]
            feature_train.append(tensor_value.tensors[0])
            
        feature_train = torch.concat(feature_train,dim = 0)
        
        #min and max on each chanel
        min_0 = torch.min(feature_train[:,0,:])
        min_1 = torch.min(feature_train[:,1,:])
        max_0 = torch.max(feature_train[:,0,:])
        max_1 = torch.max(feature_train[:,1,:])

        if max_0 == min_0 or max_1 == min_1:
            scaler_train = transforms.Normalize((0,0),(1,1))
            return train_image, test_image, scaler_train

        else:
            #create transform normalize
            scaler_train = transforms.Normalize((min_0,min_1),(max_0-min_0,max_1-min_1))
        
            #save it back to dict with normalized pixel 
            for key in train_image.keys():
                tensor_value = train_image[key]
                feature_train, label_train = tensor_value.tensors[0], tensor_value.tensors[1]
                feature_train = scaler_train(feature_train)
                tensor_value = TensorDataset(feature_train,label_train)
                train_image[key] = tensor_value
            
            for key in test_image.keys():
                tensor_value = test_image[key]
                feature_test, label_test = tensor_value.tensors[0], tensor_value.tensors[1]
                feature_test = scaler_train(feature_test)
                tensor_value = TensorDataset(feature_test,label_test)
                test_image[key] = tensor_value
                                
            return train_image, test_image, scaler_train
        
    def true_label(self, data_dict:dict):
        #function to get true label of a bearing train dict dataset 
        true_value = {}
        for key in data_dict.keys():
            label = data_dict[key].tensors[1]
            label = label[:,0]
            true_value[key] = label
        return true_value
    
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
                
    def stft(self, dict_dataset:dict, w_stft = 2560, hop = 128):
        
        for key in dict_dataset.keys():
        #for key in tqdm(dict_dataset.keys()):
            #load data
            raw_data = dict_dataset[key]
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
            
            #add it to dict:
            dict_dataset[key] = data_transform
            
        return dict_dataset

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

    def lms(self,dict_dataset,w_stft=2560,hop=256,n_mels=256):
        
        for key in dict_dataset.keys():
        #for key in tqdm(dict_dataset.keys()):
            #load data
            raw_data = dict_dataset[key]
            len_data = len(raw_data)
            raw_data, label = raw_data.tensors
            
            #features transform tensor.
            feature_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))

            for index,feature in enumerate(raw_data):
            #for index,feature in enumerate(tqdm(raw_data)):    
                #create image:
                #_, _, feature_image = stft_scipy(x=feature,fs = self.sr,nperseg=w_stft,noverlap=w_stft-hop)
                feature_image = librosa.feature.melspectrogram(y=feature.numpy(),sr=self.sr,n_fft=w_stft,hop_length=hop,n_mels=n_mels,htk = True)
                feature_image = librosa.power_to_db(feature_image)
                feature_image = torch.from_numpy(feature_image)
                
                #transform image
                feature_image = self.transform(feature_image)
                #indexing image and label
                feature_transform[index] = feature_image

            #create TensorDataset     
            data_transform = TensorDataset(feature_transform,label)
            
            #add it to dict:
            dict_dataset[key] = data_transform
            
            
        return dict_dataset
    
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
    
    def cwt(self,dict_dataset:dict,b:float = 1.5,fc:float = 1.0,scale_min:float = 2, scale_max:float = 20 ,n_scales:int=250):
        
        #load wavelet and scale
        wavelet = "cmor{}-{}".format(b,fc)
        scales = np.linspace(scale_min,scale_max,n_scales)
        
        #for key in dict_dataset.keys():
        for key in tqdm(dict_dataset.keys()):
            #load data
            raw_data = dict_dataset[key]
            len_data = len(raw_data)
            raw_data, label = raw_data.tensors
            
            #features transform tensor.
            feature_transform = torch.empty(size=(len_data,2,self.image_size,self.image_size))

            #for index,(feature_1,feature_2) in enumerate(raw_data):
            for index,(feature_1,feature_2) in enumerate(tqdm(raw_data)):    
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
            
            #add it to dict:
            dict_dataset[key] = data_transform
            
        return dict_dataset
                          
if __name__ == "__main__":  

    from timeit import default_timer
    np.random.seed (1998)
    
    start = default_timer()

    #pretrained weight from pretrained model
    weights = models.EfficientNet_B0_Weights.DEFAULT

    transform_efb0 = weights.transforms(antialias=True)

    transform_custom = transforms.Compose([
        transforms.Resize(size=(224,224),interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),
        
    ])
    
    tranformation = Transformation(transform=transform_custom)
    window_size = 25600
    
    len_index = tranformation.len_index(window_size=window_size,kind="train")
    bearing_index = tranformation.bearing_indices(window_size=window_size,kind="train")
    #print("bearing_index", bearing_index)
    #print("len_index", len_index)
    
    #print(tranformation.scaler)
    train_data = tranformation.load_raw_data(kind_data="train",window_size=window_size)
    #print("train_data", train_data)
    
    train_bearing = [12,21,22,31,32]
    val_bearing = 11
    
    #true_label = tranformation.true_label(train_data)
    #print("true_label", true_label)
    
    train_data, test_data, _ = tranformation.normalize_raw(train_val_raw_data= train_data,train_bearing=train_bearing,val_bearing=val_bearing, min_max_scaler= False)
    
    #pair_train = tranformation.create_pairs(train_data)
    #print("pair_train:", pair_train)
    
    for i in train_data.values():
        print(len(i))

    for i in test_data.values():
        print(len(i))
    #print()
    
    n_windows = tranformation.n_windows_bearing(window_size=25600)
    print("n_windows", n_windows)
    
    #test_data = tranformation.load_raw_data(kind_data="full",window_size=window_size)
    #for i in test_data.values():
    #    print(len(i))
    #train_data, test_data, _ = tranformation.normalize_raw(train_val_raw_data= train_data,test_raw_data=test_data)
    
    #feature, label = [],[]
    #for tensor_value in train_data.values():
        #print("len tensor value",len(tensor_value))
        #feature.append(tensor_value.tensors[0])
        #label.append(tensor_value.tensors[1])
    
    #feature = torch.concat(feature,dim =0)
    #label =  torch.concat(label,dim =0)
    
    #dataset = TensorDataset(feature,label)
    #print("dataset len", len(dataset))
    
    #for i in train_data.values():
    #    print(len(i))
    #    #print(i[:][0])
    #    #print()
    #    print(torch.max(i[:][0]))
    #    print(torch.min(i[:][0]))
    #    print()
    
    #for i in test_data.values():
    #    print(len(i))
    #    print(torch.max(i[:][0]))
    #    print(torch.min(i[:][0]))
    #    print()

    #transform the data
        #feature_train = scaler.transform(feature_train).reshape(len(label_train),-1,chanel)
        #feature_train = np.transpose(feature_train,(0,2,1))
        #feature_val = scaler.transform(feature_val).reshape(len(label_val),-1,chanel) 
        #feature_val = np.transpose(feature_val,(0,2,1))
        
    #print("train_data:", len(train_data))
    #test_data = tranformation.load_raw_data(kind_data="test",window_size=window_size)
    #print("test_data:", len(test_data))
    
    #print("data:", len(data))
    #val_index = np.arange(0,2803).tolist()
    #print("val_index:", len(val_index))
    #train_index = np.arange(2803,len(train_data)).tolist()
    #print("train_index:", len(train_index))
    #print("val_data:", len(val_data))
    #print("train_data:", train_data[0][0]) #
    #print("train_data:", train_data[0][1])

    #print("train_data:", val_data[0][0]) #
    #print("train_data:", val_data[0][1])
    """train_data = data[np.arange(0,2803)]
    val_data = data,[np.arange(2803,len(data))]"""
    
    """train_data_norm, test_data_norm,_ =  tranformation.normalize_raw(train_data,train_index=train_index,val_index=val_index,min_max_scaler=False,clamp=False) #,val_data 
    lms = tranformation.stft(test_data_norm)
    
    import matplotlib.pyplot as plt
    image = lms[2802][0][0]
    print("image:", image.shape)
    print("image:", image)
    plt.imshow(image,aspect="auto", interpolation="none", )#origin='lower'
    plt.colorbar()"""
    
    #full_data = tranformation.load_raw_data(kind_data="full",window_size=window_size)
    #train_data_norm, test_data_norm,_ =  tranformation.normalize_raw(train_data,test_raw_data=full_data,min_max_scaler=True,clamp=False)
    #stft = tranformation.stft(test_data_norm)
    
    #print("train_data_norm len:", len(train_data_norm))
    #print("test_data_norm len:", len(test_data_norm))
    #train_data_norm, test_data_norm =  tranformation.normalize_raw(train_data,train_index=train_index,val_index=val_index,min_max_scaler=False,best_cv=False) #,val_data 
  
    """train_stft = tranformation.stft(train_data)
    val_stft = tranformation.stft(val_data)
    train_norm, test_norm = tranformation.normalize_pixel(train_stft,val_stft)"""
      
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

    STFT = tranformation.cwt(train_data)  
    print(STFT)
    print(len(STFT))
    tensor = STFT[12]
    feature, label = tensor.tensors
    print(feature.shape)
    print(label.shape)
    print(feature[0])
    print()
    
    """STFT_test = tranformation.stft(test_data)
    train_data, test_data, _ = tranformation.normalize_pixel(STFT, STFT_test)
    
    print(train_data)
    print(len(train_data))
    tensor = train_data[12]
    feature, label = tensor.tensors
    print(feature.shape)
    print(label.shape)
    print(feature[0])"""
    
    """import matplotlib.pyplot as plt
    image = STFT[2793][0][0]
    plt.imshow(image,aspect="auto", interpolation="none", origin='lower')
    plt.colorbar()
    #plt.show()"""
    
    """import matplotlib.pyplot as plt
    image = STFT[0][0][0]
    plt.imshow(image,aspect="auto", interpolation="none")
    plt.show()"""
    
    end = default_timer()
    print(end-start)
