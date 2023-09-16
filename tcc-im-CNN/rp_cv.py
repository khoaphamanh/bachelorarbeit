from transformation import Transformation
from torchvision import transforms
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
from neptune.utils import stringify_unsupported
from joblib import Parallel, delayed
from multiprocessing import Manager

import torch
import torch.nn as nn
import torchvision
import neptune
import matplotlib.pyplot as plt
import optuna
import numpy as np
import os
import pickle
import random

class CrossValidation:
    def __init__(self,PROJECT:str,API_TOKEN:str,transformation:Transformation,K,W_SIZE,MODEL_NAME,MAE,CE,device):
        #fixparameter
        self.k = K
        self.w_size = W_SIZE
        self.transformation = transformation
        self.device = device
        self.model_name = MODEL_NAME
        self.ce = CE
        self.mae = MAE
        self.project = PROJECT
        self.api_otken = API_TOKEN
        
        #bearing inforamtion
        self.bearing_train_number = self.transformation.bearing_train
        self.bearing_test_number = self.transformation.bearing_test
        self.inverse_transform = self.transformation.inverse_transform_label
        self.eol  = self.transformation.eol
        self.last_cycle = self.transformation.last_cycle
        self.rul = self.transformation.rul
        
        #helper function
        self.normalize_raw = self.transformation.normalize_raw
        self.n_windows_bearing = self.transformation.n_windows_bearing
        self.len_index = self.transformation.len_index
        self.lc_window_index = self.transformation.lc_window_index(window_size=self.w_size)
        self.create_pairs = self.transformation.create_pairs
        self.true_label = self.transformation.true_label
        
        #saved variable:
        self.train_val_raw_data = self.load_raw_data(kind_data="train")
        self.scaler_raw_cv = Manager().dict() 
        self.scaler_pixel_cv = Manager().dict()
        self.scaler_pixel_total = Manager().dict() 
        self.dir = "model_pretrained/{}".format(self.project.split("/")[1].split("-")[0]) 
        
    def load_raw_data(self,kind_data:str):
        #function to load raw data
        raw_data = self.transformation.load_raw_data(kind_data=kind_data,window_size=self.w_size)
        return raw_data
    
    def transform_image (self, train_raw_data, test_raw_data, threshold, percentage, dimension, time_delay):
        #finction to transform raw_norm_data to image and normalize pixel it
        train_image = self.transformation.rp(train_raw_data, threshold, percentage, dimension, time_delay)
        test_image =  self.transformation.rp(test_raw_data, threshold, percentage, dimension, time_delay)
        train_image, test_image, scaler_pixel = self.transformation.normalize_pixel(train_image=train_image,test_image=test_image)
        return train_image, test_image, scaler_pixel 
    
    def t_FPT(self,cycle_sequence_dict: dict, n_FPT: int, n_period: int):
        t_FPT_dict = {}
        for key in cycle_sequence_dict.keys():
            bearing = cycle_sequence_dict[key]
            t_FPT_dict[key] = []
            for i in range(len(bearing)):
                sequence = bearing[i]
                for index in range(len(sequence)):
                    if index + n_period <= len(sequence) and sum(sequence[index:index + n_period]) > n_FPT:
                        FPT = index
                        t_FPT_dict[key].append(FPT)
                        break

                if len(t_FPT_dict[key]) < i+1:
                    t_FPT_dict[key].append(None)
                
        return t_FPT_dict 
    
    def save_scaler (self,split, scaler_raw, scaler_pixel):
        
        #function to check if saved scaler available in directory model_saved
        file_path_scaled_raw_cv = os.path.join(self.dir,"scaler_raw_cv.pkl")
        file_path_scaled_pixel_cv = os.path.join(self.dir,"scaler_pixel_cv.pkl")
                
        if not os.path.isfile(file_path_scaled_raw_cv):
            if len(self.scaler_raw_cv) < self.k: 
                self.scaler_raw_cv[split]=scaler_raw 
                
            if len(self.scaler_raw_cv) == self.k: 
                with open(file_path_scaled_raw_cv,"wb") as file:
                    pickle.dump(dict(self.scaler_raw_cv),file)
        
        self.scaler_pixel_cv[split]=scaler_pixel
        if len(self.scaler_pixel_cv) < self.k:
            self.scaler_pixel_cv[split]=scaler_pixel
                
        if len(self.scaler_pixel_cv) == self.k:
            self.scaler_pixel_total[self.trial_number] = dict(self.scaler_pixel_cv)
            if not os.path.isfile(file_path_scaled_pixel_cv):
                with open(file_path_scaled_pixel_cv,"wb") as file:
                    pickle.dump(dict(self.scaler_pixel_total),file)  
                    
            elif os.path.isfile(file_path_scaled_pixel_cv) and os.path.getsize(file_path_scaled_pixel_cv) > 0:
                with open(file_path_scaled_pixel_cv, 'rb') as file:
                    loaded_scaler_pixel = pickle.load(file)
                loaded_scaler_pixel.update(self.scaler_pixel_total)    
                
                with open(file_path_scaled_pixel_cv,"wb") as file:
                    pickle.dump(dict(loaded_scaler_pixel),file)  
 
    def load_scaler (self,split, test_raw_data:TensorDataset,w_stft, hop, clamp = False):
        #function to load a saved scaler and transform it to test_raw_data
        file_path = os.path.join(self.dir,"scaler_raw_cv.pkl")
        
        with open(file_path,"rb") as file:
            scaler_dict = pickle.load(file)
        
        #split feature, label and reshape
        feature_test, label_test = test_raw_data.tensors
        chanel = 2
        feature_test = feature_test.transpose(1,2).reshape(-1,chanel)
        
        #transform the feature
        scaler = scaler_dict[split]
        feature_test = scaler.transform(feature_test).reshape(len(label_test),-1,chanel) 
        feature_test = np.transpose(feature_test,(0,2,1))
        feature_test = torch.from_numpy(feature_test)
        
        #clamp
        if clamp  == True:
            feature_test = np.clip(feature_test,-1,1)
        
        #transform to TensorDataset
        test_data = TensorDataset(feature_test,label_test)
        
        #transform to image
        test_data = self.transformation.stft(test_data,w_stft,hop)    
        feature_test, label_test = test_data.tensors
        
        file_path = os.path.join(self.dir,"scaler_pixel_cv.pkl")
        with open(file_path,"rb") as file:
            scaler_dict = pickle.load(file)
        scaler = scaler_dict[self.best_trial_number][split]
        
        feature_test = scaler(feature_test)    
        
        return TensorDataset(feature_test,label_test)
    
    def load_model (self,drop:float,embedding_dim):
        #load model
        model = getattr(torchvision.models,self.model_name)()
        
        #change inout and output of the model
        model.features[0][0] = nn.Conv2d(INPUT,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier = nn.Sequential(
        nn.Dropout(p=drop,inplace=True),
        nn.Linear(in_features=1280, out_features=2, bias=True),
        nn.Linear(in_features=2, out_features=embedding_dim, bias=True))

        return model
    
    def score(self,y_pred,y_true):
        eps = torch.tensor(1e-7)
        Er = 100 * (y_true - y_pred) / torch.maximum(y_true,eps)
        A = torch.where(Er <= 0,torch.exp(-torch.log(torch.tensor(0.5)) * (Er / 5 )),torch.exp(torch.log(torch.tensor(0.5)) * (Er / 20 )))
        score = torch.mean(A)
        return score
    
    def RUL (self, FPT_train:dict, FPT_val:dict, neareast_neighbor_dict_val:dict):
        #check if None:
        for key in FPT_train.keys():
            if None in FPT_train[key]:
                return None
        
        for key in FPT_val.keys():
            if None in FPT_val[key]:
                return None
                
        #calculate percentage RUL
        y = {}
        for key in FPT_train.keys():
            N_train = self.n_windows_bearing(self.w_size)["train"][key]
            FPT_mean = round(np.mean(FPT_train[key]))
            indices = np.arange(0,N_train)
            y_value = (N_train-1-indices) / (N_train-1-FPT_mean)
            y_value = np.where(y_value>1,1,y_value)
            y[key] = y_value
        
        #calculate y dach
        y_dach = {}
        for key in neareast_neighbor_dict_val.keys():
            neareast_neighbor = neareast_neighbor_dict_val[key]
            y_dach_value = []
            train_keys = list(y.keys())
            FPT_mean = round(np.mean(FPT_val[key]))
            
            for kei in range(len(train_keys)):
                neareast_neighbor_tmp = neareast_neighbor[kei]
                y_dach_tmp = y[train_keys[kei]][neareast_neighbor_tmp]
                y_dach_value.append(y_dach_tmp)
            
            y_dach_key = np.mean(y_dach_value,axis=0)
            y_dach_key[0:FPT_mean] = 1
            y_dach[key] = y_dach_key
            
        #calculate rul
        rul = {}
        for key in y_dach.keys():
            FPT_mean = round(np.mean(FPT_val[key]))
            y_dach_value = y_dach[key]
            indices = np.arange(0,len(y_dach_value))
            rul_value_ts = indices+1-FPT_mean
            rul_value_ms = 1-y_dach_value
            rul_value = np.where(rul_value_ms!=0, rul_value_ts/rul_value_ms*y_dach_value,len(y_dach_value))
            rul_value[0:FPT_mean] = np.flip(np.linspace(rul_value[FPT_mean],len(y_dach_value),FPT_mean))     
            rul[key] = rul_value / len(rul_value)
            
        return rul
    
    def calulate_loss_and_score (self,rul:dict,true_label:dict):
        #check if rul None
        if rul is None:
            return 5,0
        
        #calculate the score and 
        losses = []
        scores = []
        for key in true_label.keys():
            true_value = true_label[key]
            rul_value = torch.from_numpy(rul[key]) 
            loss = self.mae(rul_value,true_value)
            rul_value = torch.clamp(rul_value,0,1)
            score = self.score(rul_value,true_value)
            losses.append(loss.item())
            scores.append(score.item())
            
        return np.mean(losses),np.mean(scores)

    def train_evaluate_loop(self,run:neptune.init_run,model:nn.Module,train_data:dict,val_data:dict,batch_size:int,epochs,optimizer:torch.optim,n_FPT:int,n_period:int,cv:bool=True):
        
        #keys of the data
        train_keys = train_data.keys()
        val_keys = val_data.keys()
        true_label_val = self.true_label(val_data)
        text_report = ""
        
        #training loop
        for iter in range(epochs):
            
            #shuffle and create pairs
            train_pairs = self.create_pairs(train_data=train_data,shuffle=True)
            val_pairs = self.create_pairs(train_data=train_data,test_data=val_data,shuffle=True)
            
            #training mode:
            model.train()
            
            #loop through each pair to select u and v from train data
            for pair in train_pairs:
                
                #select u and v from train data
                s_train = train_data[pair[0]]
                t_train = train_data[pair[1]]
                s_train,_ = s_train.tensors
                t_train,_ = t_train.tensors
                
                #print the max sequence length
                seq_len = max(len(s_train),len(t_train))
                
                #do a loop for each iteration
                for _ in range(int(np.ceil(seq_len // batch_size))):
                    
                    #loss of u and v each pair
                    loss_pair_u_train = 0
                    loss_pair_v_train = 0
                
                    #create dataloader
                    if len(s_train) > batch_size:
                        s_train = s_train[random.sample(range(len(s_train)),batch_size)]
                    else: 
                        s_train = s_train[random.sample(range(len(s_train)),len(s_train))]
                    
                    if len(t_train) > batch_size:
                        t_train = t_train[random.sample(range(len(t_train)),batch_size)]
                    else:
                        t_train = t_train[random.sample(range(len(t_train)),len(t_train))]
                        
                    #load tensors to device
                    s_train = s_train.to(self.device)
                    t_train = t_train.to(self.device)
                    
                    #forward pass u and v
                    u_train = model(s_train)
                    v_train = model(t_train)
                    
                    #loop through index of every u to find soft nearest neighbor in v
                    for index in range(len(u_train)):
                        
                        #choose u_i in u and repeat as length of v
                        u_i_train = u_train[index]
                        u_i_train = u_i_train.repeat(len(v_train),1)
                        
                        #calculate alpha, v tilde and beta
                        alpha_train = torch.softmax(-torch.sqrt(torch.sum((u_i_train-v_train)**2,dim = 1,keepdim=True)),dim = 0)
                        v_t_train = torch.sum(alpha_train*v_train,dim = 0)
                        beta_train = -torch.sqrt(torch.sum((v_t_train-u_train)**2,dim = 1)).unsqueeze(0)
                        
                        #calculate the loss
                        loss_u_train = self.ce(beta_train,torch.tensor([index]).to(self.device))
                        loss_pair_u_train = loss_pair_u_train + loss_u_train
                        
                    #normalize the loss
                    loss_pair_u_train = loss_pair_u_train / len(u_train)
                    
                    #update the gradient 
                    optimizer.zero_grad()
                    loss_pair_u_train.backward()
                    optimizer.step()

                    #forward pass u and v
                    u_train = model(s_train)
                    v_train = model(t_train)
                    
                    #loop through idnex of every v to find soft nearest neighbor in u
                    for index in range(len(v_train)):
                        
                        #choose v_i in v and repeat it as length of u
                        v_i_train = v_train[index]
                        v_i_train = v_i_train.repeat(len(u_train),1)
                        
                        #calculate alpha, v tilde and beta
                        alpha_train = torch.softmax(-torch.sqrt(torch.sum((v_i_train-u_train)**2,dim=1,keepdim=True)),dim = 0)
                        u_t_train = torch.sum(alpha_train*u_train,dim = 0)
                        beta_train = -torch.sqrt(torch.sum((u_t_train-v_train)**2,dim=1)).unsqueeze(0)
                        
                        #calculate the loss
                        loss_v_train = self.ce(beta_train,torch.tensor([index]).to(self.device))
                        loss_pair_v_train = loss_pair_v_train + loss_v_train
                    
                    #normalize the loss
                    loss_pair_v_train = loss_pair_v_train / len(v_train)

                    #update the gradient 
                    optimizer.zero_grad()
                    loss_pair_v_train.backward()
                    optimizer.step()
            
            #evaluating
            model.eval()
            with torch.inference_mode():
                
                #total loss each epoch, the sum of all pairs
                loss_total_train = 0
                
                #shuffle and create pairs
                train_pairs = self.create_pairs(train_data=train_data,shuffle=False)
                val_pairs = self.create_pairs(train_data=train_data,test_data=val_data,shuffle=False)
                
                #cycle sequence dict to store the sequence of cycle points
                cycle_sequence_dict_train = {} 
                for key in train_keys:
                    cycle_sequence_dict_train[key] = []
                    
                #accuracy dict and list to store the accuracy of cycle points
                accuracy_dict_train = {} 
                for keys in train_keys:
                    accuracy_dict_train[keys] = []
                accuracy_list_train = []
                
                #loop through each pair to select u and v from train data
                for pair in train_pairs:
                    
                    #loss of u and v each pair
                    loss_pair_u_train = 0
                    loss_pair_v_train = 0
                    
                    #sequence of u and v if cycle point
                    cycle_sequence_u_train = []
                    cycle_sequence_v_train = []
                    
                    #select u and v from train data
                    s_train = train_data[pair[0]]
                    t_train = train_data[pair[1]]
                    s_train,_ = s_train.tensors
                    t_train,_ = t_train.tensors
                    
                    #create dataloader
                    s_train = DataLoader(s_train,shuffle=False)
                    t_train = DataLoader(t_train,shuffle=False)
                    
                    #create feature map from data laoder and concat it
                    u_train = []
                    v_train = []
                    for image in s_train:
                        image = image.to(self.device)
                        feature = model(image)
                        u_train.append(feature)
                        
                    for image in t_train:
                        image = image.to(self.device)
                        feature = model(image)
                        v_train.append(feature)
                        
                    u_train = torch.concatenate(u_train,dim = 0)
                    v_train = torch.concatenate(v_train,dim = 0)
                    
                    #loop through index of every u to find soft nearest neighbor in v
                    for index in range(len(u_train)):
                        
                        #choose u_i in u and repeat as length of v
                        u_i_train = u_train[index]
                        u_i_train = u_i_train.repeat(len(v_train),1)
                        
                        #calculate alpha, v tilde and beta
                        alpha_train = torch.softmax(-torch.sqrt(torch.sum((u_i_train-v_train)**2,dim = 1,keepdim=True)),dim = 0)
                        v_t_train = torch.sum(alpha_train*v_train,dim = 0)
                        beta_train = -torch.sqrt(torch.sum((v_t_train-u_train)**2,dim = 1)).unsqueeze(0)
                        
                        #calculate the loss
                        loss_u_train = self.ce(beta_train,torch.tensor([index]).to(self.device))
                        loss_pair_u_train = loss_pair_u_train + loss_u_train
                        
                        #check if cycle consistency point
                        beta_train = torch.argmax(beta_train.ravel(),dim = 0)
                        if beta_train == index:
                            cycle_sequence_u_train.append(1)
                        else:
                            cycle_sequence_u_train.append(0)
                        
                    #save the cycle sequence in dict, save accuracy in dict and calculate the total loss pair u
                    cycle_sequence_dict_train[pair[0]].append(cycle_sequence_u_train)
                    accuracy_dict_train[pair[0]].append(np.mean(cycle_sequence_u_train)*100)
                    loss_pair_u_train = loss_pair_u_train / len(u_train)
                    
                    #loop through idnex of every v to find soft nearest neighbor in u
                    for index in range(len(v_train)):
                        
                        #choose v_i in v and repeat it as length of u
                        v_i_train = v_train[index]
                        v_i_train = v_i_train.repeat(len(u_train),1)
                        
                        #calculate alpha, v tilde and beta
                        alpha_train = torch.softmax(-torch.sqrt(torch.sum((v_i_train-u_train)**2,dim=1,keepdim=True)),dim = 0)
                        u_t_train = torch.sum(alpha_train*u_train,dim = 0)
                        beta_train = -torch.sqrt(torch.sum((u_t_train-v_train)**2,dim=1)).unsqueeze(0)
                        
                        #calculate the loss
                        loss_v_train = self.ce(beta_train,torch.tensor([index]).to(self.device))
                        loss_pair_v_train = loss_pair_v_train + loss_v_train
                        
                        #check if cycle consistency point
                        beta_train = torch.argmax(beta_train.ravel(),dim = 0)
                        if beta_train == index:
                            cycle_sequence_v_train.append(1)
                        else:
                            cycle_sequence_v_train.append(0)
                            
                    #save the cycle consistency in dict, save accuracy in dict and calculate the loss pair v
                    cycle_sequence_dict_train[pair[1]].append(cycle_sequence_v_train)
                    accuracy_dict_train[pair[1]].append(np.mean(cycle_sequence_v_train)*100)
                    loss_pair_v_train = loss_pair_v_train / len(v_train)
                    
                    #calculate the loss total train every pair
                    loss_total_train = loss_total_train + loss_pair_u_train.item() + loss_pair_v_train.item() # 
                
                #calculate the FPT
                FPT_train = self.t_FPT(cycle_sequence_dict=cycle_sequence_dict_train,n_FPT=n_FPT, n_period=n_period)
                
                #calculate the accuracy
                for key in train_keys:
                    accuracy_list_train.append(np.mean(accuracy_dict_train[key]))
                accuracy_train = np.mean(accuracy_list_train)
                
                #caculate the loss of all pairs
                loss_total_train = loss_total_train / len(train_pairs)
            
                #loss total of each epoch, the sum of all pairs
                loss_total_val = 0
                
                #cycle sequence dict to store the sequence of cycle points
                cycle_sequence_dict_val = {}
                for key in val_keys:
                    cycle_sequence_dict_val[key] = []
                                    
                #accuracy dict and list to store the accuracy of cycle points
                accuracy_dict_val = {}
                for key in val_keys:
                    accuracy_dict_val[key] = []
                accuracy_list_val = []
                
                #nearest neighbor val
                nearest_neighbor_dict_val = {}
                for keys in val_keys:
                    nearest_neighbor_dict_val[keys] = []
                    
                #loop through each pair to select u and v from val data
                for pair in val_pairs:
                    
                    #loss of u and v each pair
                    loss_pair_u_val = 0
                    
                    #sequence of u and v if cycle point
                    cycle_sequence_u_val = []
                    
                    #nearest nieghtbor list
                    nearest_neighbor_u_val = []
                    
                    #select u and v from val data
                    s_val = val_data[pair[0]]
                    t_val = train_data[pair[1]]
                    s_val,_ = s_val.tensors
                    t_val,_ = t_val.tensors
                    
                    #create dataloader:
                    s_val = DataLoader(s_val, shuffle=False)
                    t_val = DataLoader(t_val, shuffle=False)
                    
                    #create feature mao from data loader and concat it.
                    u_val = []
                    v_val = []
                    for image in s_val:
                        image = image.to(self.device)
                        feature = model(image)
                        u_val.append(feature)
                    
                    for image in t_val:
                        image = image.to(self.device)
                        feature = model(image)
                        v_val.append(feature)
                    
                    u_val = torch.concatenate(u_val, dim =0)
                    v_val = torch.concatenate(v_val, dim = 0)
                    
                    #loop through index of every u to find soft nearest neighbor in v
                    for index in range(len(u_val)):
                        
                        #choose u_i in u and repeat as length of v
                        u_i_val = u_val[index]
                        u_i_val = u_i_val.repeat(len(v_val),1)
                        
                        #calculate the alpha, v tilde and beta
                        alpha_val = torch.softmax(-torch.sqrt(torch.sum((u_i_val-v_val)**2,dim = 1,keepdim=True)),dim = 0)
                        v_t_val = torch.sum(alpha_val*v_val, dim = 0)
                        beta_val = -torch.sqrt(torch.sum((v_t_val-u_val)**2,dim = 1)).unsqueeze(0)
                        
                        #calculate the loss
                        loss_u_val = self.ce(beta_val,torch.tensor([index]).to(self.device))
                        loss_pair_u_val = loss_pair_u_val + loss_u_val
                        
                        #find the nearest neighbor
                        alpha_val = torch.argmax(alpha_val.ravel(),dim = 0)
                        nearest_neighbor_u_val.append(alpha_val.item())
                        
                        #check if cycle consistency point
                        beta_val = torch.argmax(beta_val.ravel(),dim = 0)
                        if beta_val == index:
                            cycle_sequence_u_val.append(1)
                        else:
                            cycle_sequence_u_val.append(0)

                    #save the cycle sequence in dict, save accuracy in dict and calculate the total loss pair u
                    cycle_sequence_dict_val[pair[0]].append(cycle_sequence_u_val)
                    accuracy_dict_val[pair[0]].append(np.mean(cycle_sequence_u_val)*100)
                    loss_pair_u_val = loss_pair_u_val / len(u_val)
                    nearest_neighbor_dict_val[pair[0]].append(nearest_neighbor_u_val) 
                    
                    #calculate the loss total val
                    loss_total_val = loss_total_val + loss_pair_u_val.item()
                
                #calculate the FPT
                FPT_val = self.t_FPT(cycle_sequence_dict=cycle_sequence_dict_val,n_FPT=n_FPT,n_period=n_period)
                
                #calculate the accuracy
                for key in val_keys:
                    accuracy_list_val.append(np.mean(accuracy_dict_val[key]))
                accuracy_val = np.mean(accuracy_list_val)
                
                #calculate the loss of all pairs
                loss_total_val = loss_total_val / len(val_pairs)
            
            #calculate the loss and score:
            #if iter == 0:
            #    FPT_train = {12: [20, 0,0,10], 21: [20, 10,5,0], 22: [30, 30,0,0],31:[10, 35,0,0],32:[0, 0,0,0]}
            #    FPT_val =  {11: [0, 0, 5,5]}
            #   nearest_neighbor_dict_val = {11: [np.sort(np.random.randint(0,87,(280,))),np.sort(np.random.randint(0,91,(280,))),np.sort(np.random.randint(0,79,(280,))),np.sort(np.random.randint(0,51,(280,))),np.sort(np.random.randint(0,163,(280,))) ]}
                
            rul_val = self.RUL(FPT_train=FPT_train,FPT_val=FPT_val,neareast_neighbor_dict_val=nearest_neighbor_dict_val)
            loss_val, score_val = self.calulate_loss_and_score(rul_val,true_label_val)
            
            #plot the rul true and rul predict
            fig_val = self.plot_rul(rul=rul_val,true_label=true_label_val,FPT_val=FPT_val,loss_val=loss_val,score_val=score_val,iter=iter)
            run["images/val"].append(fig_val,step=iter)
            plt.close()
            
            #save the metrics
            metric = {"loss tcc train":loss_total_train, "loss tcc val":loss_total_val, "accuracy train": accuracy_train, "accuracy val":accuracy_val, "loss val":loss_val,"score val": score_val}
            run["metrics"].append(metric,step=iter)  
            train_text = "epoch {} loss train {} accuracy train {} FPT train {} \n".format(iter,loss_total_train,accuracy_train,cycle_sequence_dict_train)  
            test_text = "epoch {} loss val {} accuracy val {} FPT val {} \n".format(iter,loss_total_val,accuracy_val,cycle_sequence_dict_val)
            text_report =   train_text + text_report  + test_text
            
            #check pruned:
            if N_JOBS_CV == 1:
                self.trial.report(loss_val,iter + self.split*epochs) 
                if self.trial.should_prune():
                    run["status"] = "pruned"
                    run.stop()
                    raise optuna.exceptions.TrialPruned()
                
        run["text"] = text_report
        
        #save the model
        model_path = os.path.join(self.dir,"t_{}_s_{}.pth".format(self.trial.number,self.split)) 
        torch.save(model.state_dict(),model_path)
            
        return loss_val, score_val
        
    def cv_load_data_parallel (self, threshold, percentage, dimension, time_delay,val_bearing):
        
        #train index and bearing train, val
        split, val_bearing = val_bearing
        train_bearing = [i for i in self.bearing_train_number if i != val_bearing]
        
        #create train_data, val_data and normalize it
        train_raw_data, val_raw_data, scaler_raw = self.normalize_raw(train_val_raw_data=self.train_val_raw_data,train_bearing=train_bearing,val_bearing=val_bearing,min_max_scaler=MIN_MAX_SCALER,clamp=CLAMP)
        
        #transform raw data to image
        train_data, val_data, scaler_pixel = self.transform_image(train_raw_data=train_raw_data,test_raw_data=val_raw_data, threshold=threshold, percentage=percentage, dimension=dimension, time_delay = time_delay)
        
        #save the scaler:
        self.save_scaler(split=split,scaler_raw = scaler_raw, scaler_pixel = scaler_pixel)
        
        return (train_data, val_data) 
    
    def cv_train_eval_parallel (self,drop:float, embedding_dim:int, batch_size:int, optimizer_name:str,lr:float,epochs:int,weight_decay:float, n_FPT:int, n_period:int, params_load):
        
        #params from load data parallel
        split, (train_data, val_data) = params_load
        self.split = split
        
        #load mode
        model = self.load_model(drop=drop,embedding_dim=embedding_dim)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        
        #optimizer
        optimizer = getattr(torch.optim,optimizer_name)(params = model.parameters(), lr = lr, weight_decay = weight_decay)
        
        #create run
        run = neptune.init_run(project=self.project, api_token=self.api_otken)
        run_name = "trial_{}_split_{}".format(self.trial_number,split)
        run["name"] = run_name
        
        #log parameters and transform images
        hyperparameters = self.trial.params     
        run["hyperparameters"] = stringify_unsupported(hyperparameters) 
        run["fix_parameters"] =  stringify_unsupported(fix_parameters)      
        run["fix_interval"] = stringify_unsupported(fix_interval) 
        
        #training loop
        loss_val_this_split, score_val_this_split = self.train_evaluate_loop(run=run,model=model,train_data=train_data,val_data=val_data,batch_size=batch_size,epochs=epochs,optimizer=optimizer,n_FPT=n_FPT,n_period=n_period)
        
        #tracking split score
        print("Split {}, loss val {:.3f}, score val {:.3f}".format(split,loss_val_this_split,score_val_this_split))    
    
        run["status"] = "finished"
        run["runing_time"] = run["sys/running_time"]
        run.stop()
        
        return loss_val_this_split, score_val_this_split
             
    def cross_validation(self,trial:optuna.trial.Trial,drop:float, embedding_dim :int, threshold, percentage, dimension, time_delay,optimizer_name:str,lr:float,batch_size:int,epochs:int,weight_decay:float,n_FPT:int, n_period:int):
        
        #trial variable
        self.trial = trial
        self.trial_number = trial.number
         
        #params from load data 
        load_data = Parallel(n_jobs=N_JOBS_LOAD_DATA)(delayed(self.cv_load_data_parallel)(threshold, percentage, dimension, time_delay,val_bearing) for val_bearing in enumerate(self.bearing_train_number))
        
        #params from cross validation train and eval 
        result = Parallel(n_jobs = N_JOBS_CV)(delayed(self.cv_train_eval_parallel)(drop,embedding_dim,batch_size,optimizer_name,lr, epochs, weight_decay, n_FPT, n_period, params_load) for params_load in enumerate(load_data))
        result = np.array(result)
        loss_val, score_val =  result[:,0].mean(),result[:,1].mean()
    
        #tracking split score
        print("trial {}, loss train mean {:.3f}, score train mean {:.3f}\n\n".format(trial.number,loss_val,score_val))  
    
        return loss_val
      
    def visualize_transform_image(self,train_val_data):
        
        bearing = int(train_val_data[0][1][1])
        fig,ax_fig = plt.subplots(nrows=4,ncols=4,figsize = (14,10))
        axes = [item for sublist in ax_fig for item in sublist]
        if W_SIZE == 2560:
            index =[0, 421, 841, 1261, 1682, 2102, 2522, 2802]
        elif W_SIZE == 25600:
            index = [0, 399, 798, 1197, 1596, 1995, 2394, 2793]
        
        for i in range(8):
    
            idx = index[i]
            ax1 = axes[i*2]
            ax2 = axes[i*2+1]

            #load image
            images_transform = train_val_data[idx][0]
            label = train_val_data[idx][1][0] 

            #axes 1 hor:
            image = ax1.imshow(images_transform[0].squeeze())
            ax1.set_title("chanel 0, label {:.2f}, hor. Acc".format(label))
            ax1.axis("off")
            image.set_clim(vmin=0,vmax=1)
            plt.colorbar(image,ax=ax1)
            
            #axes 2 ver:
            image = ax2.imshow(images_transform[1].squeeze())
            ax2.set_title("chanel 1, label {:.2f}, ver. Acc".format(label))
            ax2.axis("off")
            image.set_clim(vmin=0,vmax=1)
            plt.colorbar(image,ax=ax2)
            
        fig.suptitle("{} transformed, image size {}, bearing {}".format(METHOD,tuple(images_transform.shape),bearing))
        fig.subplots_adjust(bottom=0.03)
        return fig

    def cv_eval_final_loop (self,model:torch.nn,test_data:TensorDataset,batch_size,params):
        #function to transform test_raw_data with the saved scaler from 
        split, file = params
        test_data = test_data[split]
        
        #create test_loader
        test_loader = self.data_loader(data=test_data,batch_size=batch_size)
        
        #tracking test
        tracking_test = {}
        for bearing in self.bearing_test_number:
            tracking_test.setdefault(bearing,[[],[]])
        
        #load model
        state_dict = torch.load(os.path.join(self.dir,file),map_location=self.device)
        
        # Check the number of CUDA devices
        if torch.cuda.device_count() > 1:
            # Modify keys for multi-GPU training
            modified_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith("module."):
                    modified_state_dict["module." + key] = value
                else:
                    modified_state_dict[key] = value
        else:
            # Modify keys for single-GPU training
            modified_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("module."):
                    modified_state_dict[key[7:]] = value
                else:
                    modified_state_dict[key] = value

        # Load the modified state dictionary into your model
        model.load_state_dict(modified_state_dict)
            
        model.eval()                    
        with torch.inference_mode():
            for X_test,y_test in test_loader:
            
                #load bering
                bearing_test = y_test[:,1].int()
                
                #load data to device
                X_test = X_test.to(self.device)
                y_test = y_test[:,0].to(self.device)
                
                #forward pass
                y_test_pred = model(X_test).ravel()
                y_test_pred = y_test_pred.clamp(0,1)

                #indexing tracking
                for bear in self.bearing_test_number:
                    tracking_test[bear][0] = tracking_test[bear][0] + y_test_pred[bearing_test==bear].tolist()
                    tracking_test[bear][1] = tracking_test[bear][1] + y_test[bearing_test==bear].tolist()

        #sort the tracking dict
        tracking_test = self.sort_trackinkg_cv(tracking_dict=tracking_test)
        return tracking_test
    
    def best_cv_final (self, run:neptune.init_run,model:torch.nn ,best_trial_number:int,full_raw_data:TensorDataset,batch_size,w_stft,hop):
        
        #tracking test
        test_pred = {}
        test_true = {}
        tracking_test_final = {}
        for bearing in self.bearing_test_number:
            test_pred.setdefault(bearing,np.empty(shape=(self.k,self.n_windows_bearing(window_size=self.w_size)["full"][bearing])))
            test_true.setdefault(bearing,np.empty(shape=(self.n_windows_bearing(window_size=self.w_size)["full"][bearing],)))
            tracking_test_final.setdefault(bearing,[[],[]])
                      
        #metrics
        score_test = 0
        loss_test = 0   

        #load data parallel:
        test_data = Parallel(n_jobs=N_JOBS_LOAD_DATA_BEST_CV)(delayed(self.load_scaler)(split=split,test_raw_data=full_raw_data,w_stft=w_stft,hop=hop,clamp= CLAMP) for split in range(0,self.k))
        
        #load the result from a eval loop
        pre_trained_models = [file for file in sorted (os.listdir(self.dir)) if ("final" not in file) and file.startswith("t_{}_".format(best_trial_number))]
        result_eval_best_cv = Parallel(n_jobs=N_JOBS_BEST_CV)(delayed(self.cv_eval_final_loop)(model,test_data,batch_size,params) for params in enumerate(pre_trained_models))
        
        #calcualate the mean of the rul predict
        for split in range(len(result_eval_best_cv)):
            tracking_dict = result_eval_best_cv[split]
            for bear in self.bearing_test_number:
                test_pred[bear][split] = tracking_dict[bear][0]
                if split == 0:
                    test_true[bear] = tracking_dict[bear][1]
                    
        #calculate the mean of split prediction and indexing dict   
        for bear in self.bearing_test_number:
            test_pred[bear] = test_pred[bear].mean(axis = 0)
            tracking_test_final[bear][0] = test_pred[bear].tolist()
            tracking_test_final[bear][1] = test_true[bear]
            
        #calculate the mean and loss and score
        pred_value_all = np.empty(0,)
        true_value_all = np.empty(0,)
        for bear in self.bearing_test_number:
            pred_value_all = np.concatenate((pred_value_all,test_pred[bear]))
            true_value_all = np.concatenate((true_value_all,test_true[bear]))
            
        loss_test = self.loss(torch.from_numpy(pred_value_all),torch.from_numpy(true_value_all))
        score_test = self.score(torch.from_numpy(pred_value_all),torch.from_numpy(true_value_all))
                
        fig_rul_test, score_final_split = self.plot_rul(kind_data="test",tracking_dict=tracking_test_final)
        run["images/RUL_test_cv"].append(fig_rul_test)
        plt.close()
                            
        #log metrics 
        metric = {"score test":score_test,"loss test":loss_test, "score final":score_final_split} 
        run["metric"].append(metric)
        
        #texting
        text = "Tracking result from best trial in Cross Validation\n"
        text = text + "loss test {:.2f}, score test {:.2f}\n".format(loss_test,score_test)
        rul_pred_last = []
        rul_true_last = []
        
        for bear in self.bearing_test_number:
            rul_pred_bearing_last = self.inverse_transform(bearing=bear,label = np.array(tracking_test_final[bear][0][self.lc_window_index[bear]]))
            rul_true_bearing_last = self.inverse_transform(bearing=bear,label = np.array(tracking_test_final[bear][1][self.lc_window_index[bear]])) 
            rul_pred_last.append(rul_pred_bearing_last)
            rul_true_last.append(rul_true_bearing_last)
            text = text + "Bearing {}, RUL pred {}, RUl true {}\n".format(bear,round(rul_pred_bearing_last),round(rul_true_bearing_last))
        
        score_final = self.score(torch.tensor(rul_pred_last),torch.tensor(rul_true_last))
        loss_final = self.loss(torch.tensor(rul_pred_last),torch.tensor(rul_true_last))
        text = text + "score final {:.2f}\n".format(score_final)
        text = text + "loss final {:.2f}\n".format(loss_final)
        
        #log text
        run["result"] = text
        
        return text

    def sort_trackinkg_cv(self,tracking_dict:dict):
        
        for bearing in self.bearing_test_number:
            #sort the index of the rul true
            sort_index = np.argsort(tracking_dict[bearing][1]) #low to high
            rul_true = np.array(tracking_dict[bearing][1])[np.flip(sort_index)]
            rul_pred = np.array(tracking_dict[bearing][0])[np.flip(sort_index)]
            
            tracking_dict[bearing][0] = rul_pred.tolist()
            tracking_dict[bearing][1] = rul_true.tolist()
            
        return tracking_dict
    
    def plot_rul(self, rul:dict, true_label:dict, FPT_val:dict=None, loss_val:float=None, score_val:float=None, iter = None, cv:bool = True):
        
        #check if cv:
        if cv:
            #create figure
            fig = plt.figure(figsize=(10,10))
            bearing_val = list(true_label.keys())[0]
            true_value = true_label[bearing_val].numpy()
            plt.suptitle("RUL val bearing {} split {} epoch {}".format(bearing_val,self.split,iter))
            
            #plot the rul val
            time_stamp = np.flip(true_value)
            plt.plot(time_stamp,true_value)
            
            if rul is None:
                loss_val, score_val = "None", "None"
                plt.legend(["RUL true"])
                plt.title("loss {} score {}".format(loss_val,score_val))
            else:
                rul_value = rul[bearing_val]
                rul_value = np.clip(rul_value,0,1)
                FPT = round(np.mean(FPT_val[bearing_val]))
                plt.plot(time_stamp,rul_value)
                plt.scatter(time_stamp[FPT],true_value[FPT])
                plt.legend(["RUL true", "RUL predict","FPT"])
                plt.title("loss {:.3f} score {:.3f}".format(loss_val,score_val))
                
            plt.xlabel("Timestamp")
            plt.ylabel("Scaled RUL")
            
        return fig 
    
    def rename_best_cv_file(self, best_trial_number):
        for file in sorted(os.listdir(self.dir)):
            if (file.startswith("t_{}_".format(best_trial_number))) and ("final" not in file) and ("best" not in file):
                old_name = os.path.join(self.dir,file)
                new_name = os.path.join(self.dir,file.replace(".pth","_best.pth"))
                os.rename(old_name,new_name)
                                        
    def final_test (self,best_params:dict,best_trial_number):
        
        #load hyperparamters
        w_stft, hop = best_params["w_stft"], best_params["hop"]
        drop, optimizer, lr, batch_size, epochs = best_params["drop"], best_params["optimizer"], best_params["lr"], best_params["batch_size"], best_params["epochs"]
        
        self.best_trial_number = best_trial_number
        
        #load mode
        model = self.load_model(drop=drop)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        
        #load data
        full_data_raw = self.load_raw_data(kind_data="full")
        
        #normalize
        train_data, full_data, _ = self.normalize_raw(train_val_raw_data=self.train_val_raw_data,test_raw_data=full_data_raw,min_max_scaler=MIN_MAX_SCALER, clamp=CLAMP)
        
        #transform data to images
        train_data, full_data, _ = self.transform_image(train_raw_data=train_data,test_raw_data=full_data,w_stft=w_stft,hop=hop)
        
        #turn data to dataloader
        train_loader = self.data_loader(data=train_data,batch_size=batch_size)
        full_loader = self.data_loader(data=full_data,batch_size=batch_size)
        
        #create run
        run = neptune.init_run(project=self.project, api_token=self.api_otken)
        run_name = "trial_best_no_{}".format(best_trial_number)
        run["name"] = run_name
        
        #log parameters and transform images
        run["hyperparameters"] = best_params
        run["fix_parameters"] = stringify_unsupported(fix_parameters)      
        run["fix_interval"] = stringify_unsupported(fix_interval) 
        
        #tracking model structure and transform image
        model_summary = summary(model).__repr__()
        run["summary"] = model_summary
        model_structure = model.__repr__()
        run["structure"] = model_structure
        
        fig_transform = self.visualize_transform_image(train_val_data=train_data)
        run["images/transform"] =fig_transform
        plt.close()
        
        #optimizer and loss
        optimizer = getattr(torch.optim, optimizer)(params = model.parameters(), lr = lr )
        
        #training, testing model
        loss_train, score_train, loss_test, score_test,  score_final = self.train_evaluate_loop(run=run,model=model,train_data=train_loader,val_data=full_loader,epochs=epochs,optimizer=optimizer,cv=False)   
        
        #finish
        run["status"] = "finished"
        run["runing_time"] = run["sys/running_time"]
        run.stop()
        
        #create run
        run = neptune.init_run(project=self.project, api_token=self.api_otken)
        run_name = "trial_best_cv_no_{}".format(best_trial_number)
        run["name"] = run_name
        
        #log parameters and transform images
        run["hyperparameters"] = best_params
        run["fix_parameters"] = stringify_unsupported(fix_parameters)      
        run["fix_interval"] = stringify_unsupported(fix_interval)  
        
        #tracking model structure and transform image
        model_summary = summary(model).__repr__()
        run["summary"] = model_summary
        model_structure = model.__repr__()
        run["structure"] = model_structure
        
        #final best 6 cv
        text = self.best_cv_final(run=run,model=model,best_trial_number=best_trial_number,full_raw_data=full_data_raw,batch_size=batch_size,w_stft=w_stft,hop=hop)
        print(text)

        #finish
        run["status"] = "finished"
        run["runing_time"] = run["sys/running_time"]
        run.stop()
        
        #rename best cv file
        self.rename_best_cv_file(best_trial_number=best_trial_number)
        
        return loss_train, score_train, loss_test, score_test, score_final        
                               
#creat objective trial optuna
def objective(trial:optuna.trial.Trial):
    
    #load hyperparameter
    threshold = trial.suggest_categorical(name="threshold",choices=THRESHOLD)
    if threshold == "float":
        threshold = trial.suggest_float(name="threshold_float",low=THRESHOLD_FLOAT[0],high=THRESHOLD_FLOAT[1],log=THRESHOLD_FLOAT[2])
    percentage = trial.suggest_int(name="percentage",low=PERCENTAGE[0],high=PERCENTAGE[1],step=PERCENTAGE[2])
    dimension = trial.suggest_int(name="dimension",low=DIMENSION[0],high=DIMENSION[1],step=DIMENSION[2])
    time_delay = trial.suggest_int(name="time_delay",low=TIME_DELAY[0],high=TIME_DELAY[1],step=TIME_DELAY[2])
    
    #hyperparameter model
    drop = trial.suggest_float(name="drop",low=DROP[0],high=DROP[1],log=DROP[2])
    optimizer_name = trial.suggest_categorical(name="optimizer",choices=OPTIMIZER)
    lr = trial.suggest_float(name="lr",low=LR[0],high=LR[1],log=LR[2])
    weight_decay = trial.suggest_float(name="weight_decay",low=WEIGHT_DECAY[0],high=WEIGHT_DECAY[1],log=WEIGHT_DECAY[2])
    embedding_dim = trial.suggest_int(name="embedding_dim",low=EMBEDDING_DIM[0],high=EMBEDDING_DIM[1],step=EMBEDDING_DIM[2])
    n_FPT = trial.suggest_int(name="n_FPT",low=N_FPT[0],high=N_FPT[1],step=N_FPT[2])
    n_period = trial.suggest_int(name="n_period",low=n_FPT+1,high=N_PERIOD[1],step=N_PERIOD[2])
    
    #fix hyperparameter
    batch_size = trial.suggest_categorical(name="batch_size",choices=BATCH_SIZE)
    epochs = trial.suggest_categorical(name="epochs",choices=EPOCHS)
    
    #cross validation
    loss_val = cross.cross_validation(trial=trial,drop=drop,threshold=threshold,percentage=percentage,dimension=dimension,time_delay=time_delay,optimizer_name=optimizer_name,lr=lr,batch_size=batch_size,epochs=epochs,weight_decay=weight_decay,embedding_dim=embedding_dim,n_FPT=n_FPT,n_period=n_period)
    
    return loss_val

#project 
PROJECT = "ba-final/rp-3"
API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ=="
METHOD = "RP"

N_JOBS_LOAD_DATA = -1
N_JOBS_CV = 1
N_JOBS_LOAD_DATA_BEST_CV = 1
N_JOBS_BEST_CV = 1

MIN_MAX_SCALER = True
CLAMP = False

#Fixparameters               
SEED = 1998
INPUT = 2
K = 6
TOTAL_TRIALS = 100
W_SIZE = 25600
MODEL_NAME = "efficientnet_b0"
BATCH_SIZE = [100]
EPOCHS = [50]
n_trials = 1

fix_parameters = {"k":K,"seed":SEED,"model_name":MODEL_NAME,"w_size":W_SIZE,"n_trials":TOTAL_TRIALS,"batch_size":BATCH_SIZE,"epochs":EPOCHS}

#interval 
THRESHOLD = (None,"float","distance","point")
THRESHOLD_FLOAT = (0.005,0.5,True)
PERCENTAGE = (5,80,5)
DIMENSION = (1,1,1)
TIME_DELAY = (1,1,1)

EMBEDDING_DIM = (32,720,8)
N_FPT = (5,10,1)
N_PERIOD = (0,25,1)
DROP = (0.1,0.8,True)
OPTIMIZER = ("Adam","SGD")
LR = (0.00001,0.2,True)
WEIGHT_DECAY = (0.00001,0.1,True)
fix_interval = {"threshold":THRESHOLD,"threshold_float":THRESHOLD_FLOAT,"percentage":PERCENTAGE,"dimension":DIMENSION,"time_delay":TIME_DELAY,"optmizer":OPTIMIZER,"learning_rata":LR,"weight_decay":WEIGHT_DECAY,"embedding_dim":EMBEDDING_DIM, "n_FPT":N_FPT,"n_period":N_PERIOD}

#check device
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("\n")

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
#load transform 
transform_custom = transforms.Compose([
    transforms.Resize(size=(224,224),interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),
])
transformation = Transformation(transform=transform_custom)
mae = nn.L1Loss()
ce = nn.CrossEntropyLoss()

#cross validation
cross = CrossValidation(PROJECT=PROJECT,API_TOKEN=API_TOKEN,transformation=transformation,K = K,W_SIZE=W_SIZE,MODEL_NAME=MODEL_NAME,CE=ce,MAE=mae,device=device)
dir_name = cross.dir
os.makedirs(dir_name,exist_ok=True)

#create study
storage_path = os.path.join(dir_name,"{}.db".format(METHOD))
storage_name = "sqlite:///{}".format(storage_path)

if not os.path.isfile(storage_path):
    study = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=SEED),study_name=METHOD,storage=storage_name,pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=4))
    study.optimize(objective,n_trials=n_trials)    
    
else:
    study = optuna.load_study(study_name=METHOD, storage=storage_name,sampler=optuna.samplers.TPESampler(seed = SEED),pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=4))
    
    #some information of the last trials and all runned trials
    trials_last = study.trials[-1]
    trials_status = trials_last.state
    trials_params = trials_last.params
    trials_number = trials_last.number
    pruned_trials = len(study.get_trials(deepcopy=False,states=[optuna.trial.TrialState.PRUNED]))
    complete_trials = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
    running_trials = len(study.get_trials(deepcopy=False,states=[optuna.trial.TrialState.RUNNING]))
    
    #check if not run enough trials
    if os.path.isfile(storage_path) and (pruned_trials+complete_trials) < TOTAL_TRIALS:
        #check of a running or fail trials
        if trials_status == optuna.trial.TrialState.RUNNING or trials_status == optuna.trial.TrialState.FAIL:
            #trial deque
            study.enqueue_trial(trials_params)
                
        #load the previous study
        study.optimize(objective, n_trials=n_trials)
        
    else: 
        print("Study statistics: ")
        print("  Number of finished trials: ", pruned_trials + complete_trials)
        print("  Number of pruned trials: ", pruned_trials)
        print("  Number of complete trials: ", complete_trials)

        best_trial_score = study.best_trial
        best_trial_params = best_trial_score.params

        print("  Value: ", best_trial_score.value)

        print("  Params: ")
        for key, value in best_trial_params.items():
            print("    {}: {}".format(key, value))
    
        #best_trial_params = {'w_stft': 4224, 'hop': 288, 'drop': 0.11783486538870204, 'optimizer': 'SGD', 'lr': 0.0059512475955207835, 'batch_size': 32, 'epochs': 1,"weight_decay":0.0023062583173487318}
        print("final test best model")
        loss_train, score_train, loss_test, score_test, score_final = cross.final_test(best_trial_params,best_trial_score.number) #22best_trial_score.number

        #merge all the result text file
        transformation.merge_result(METHOD)
        