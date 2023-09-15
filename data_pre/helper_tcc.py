import pandas as pd
import os
import numpy as np

class DataPreprocessing:
    def __init__(self,kind:str):
        self.kind = kind
        self.cwd = os.getcwd()
        self.data_dir = os.path.join(os.path.dirname(os.getcwd()),self.kind)
        self.sr = 25600
        self.bearing_number = {"train":[11,12,21,22,31,32], "test":[13,14,15,16,17,23,24,25,26,27,33],"full":[13,14,15,16,17,23,24,25,26,27,33]}
        self.hop = 25600
        self.window_size_stft = 25600
        self.step = 2560

    def load_name_file(self):
        name_file = sorted(os.listdir(self.data_dir))
        if ".DS_Store" in name_file:
            name_file.remove(".DS_Store")
        return name_file

    def load_data(self,bearing:int):
        name_file = "{}_{}.csv".format("bearing",bearing)
        path_file = os.path.join(self.data_dir,name_file)
        df = pd.read_csv(path_file)
        return df     
    
    def n_samples(self,kind:str, bearing = None):
        #num samples of train and test data
        num_samples = {"train":{11: 7175680, 12: 2229760, 21: 2332160, 22: 2040320, 31: 1318400, 32: 4190720}, 
                       "test":{13: 4613120, 14: 2915840, 15: 5893120, 16: 5893120, 17: 3845120, 23: 3077120,24: 1566720, 25: 5125120, 26: 1464320, 27: 440320, 33: 901120},
                       "full" : {13: 6080000, 14: 3655680, 15: 6305280, 16: 6266880, 17: 5783040, 23: 5004800, 24: 1922560, 25: 5916160, 26: 1794560, 27: 588800, 33: 1111040}}
        return num_samples[kind][bearing]
    
    def load_array(self, bearing:int):
        name_file = "array_{}.npy".format(bearing)
        file_acc = os.path.join(self.data_dir, name_file)
        array_data = np.load(file_acc)
        return array_data.T  
         
    def eol (self,bearing:int):
        end_of_life = {11:28020,12:8700,21:9100,22:7960,31:5140,32:16360,13:23740,14:14270,15:24620,16:24470,17:22580,23:19540,24:7500,25:23100,26:7000,27:2290,33:4330}
        if bearing == None:
            return end_of_life
        elif type(bearing) == int:
            return end_of_life[bearing]

    def rul(self,bearing=None):
        remain_useful_life = {13:5730,14:2890,15:1610,16:1460,17:7570,23:7530,24:1390,25:3090,26:1290,27:580,33:820}
        if bearing == None:
            return remain_useful_life
        elif bearing in [13,14,15,16,17,23,24,25,26,27,33]:
            return remain_useful_life[bearing]
        else:
            return None
        
    def last_cycle(self,bearing = None):
        last_cycle = {13: 18010, 14: 11380, 15: 23010, 16: 23010, 17: 15010, 23: 12010, 24: 6110, 25: 20010, 26: 5710, 27: 1710, 33: 3510}
        if bearing == None:
            return last_cycle
        elif bearing in [13,14,15,16,17,23,24,25,26,27,33]:
            return last_cycle[bearing]
        else:
            return None
    
    def n_windows (self,window_size,kind : str):
        #calculae the number of windows in each bearing, train and test data an        
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
        
if __name__ == "__main__":

    from timeit import default_timer
    start = default_timer()

    data = DataPreprocessing("train")

    name_file = data.load_name_file()   
    #print(name_file)

    w_size = data.hop
    #print(w_size)

    df_data = data.load_data(12)
    #print(df_data)    

    data_array = data.load_array(11)
    print(data_array)
    
    eol_train = data.eol(14)
    #print(eol_train)
    
    rul_test = data.rul(14)
    #print(rul_test)
    
    n_samples = data.n_samples(kind="train",bearing=11)
    #print(n_samples)

    """sum = 0
    array_index_len = []
    window_size = 25600
    for bearing in data.bearing_number["test"]:
        array_index_len.append(sum)
        n_windows = data.n_windows(window_size,bearing)
        #print(n_windows)
        sum = sum + n_windows
    array_index_len.append(sum)
    print(array_index_len)
    #print(data.n_windows("train"))"""

    """train_norm = load_data("train","normalize")
    #print(train_norm[0])

    test_norm = load_data("test","normalize",13)
    #print(train_norm[0])

    train_stand = load_data("train","standardize")
    #print(train_norm[0])

    test_stand = load_data("test","standardize",13)
    #print(train_norm[0])"""

    window_size = 25600
    n_windows = data.n_windows(window_size=window_size,kind="full")
    print("n_windows:", n_windows)
    
    n_windows = data.n_windows(window_size=window_size,kind="train")
    print("n_windows:", n_windows)

    n_windows = data.n_windows_bearing(window_size=window_size)
    print("n_windows:", n_windows)
    
    len_indexx = data.len_index(window_size=window_size,kind="train")
    print("len_indexx:", len_indexx)
    
    len_indexx = data.len_index(window_size=window_size,kind="full")
    print("len_indexx:", len_indexx)
    
    len_indexx = data.len_index(window_size=window_size,kind="test")
    print("len_indexx:", len_indexx)
    
    test_full_indexx = data.test_full_index(window_size=window_size)
    print("test_full_indexx:", len(test_full_indexx))
    
    lc_index = data.lc_index(window_size=window_size)
    print("lc_index:", lc_index)
    
    end = default_timer()
    print(end-start)