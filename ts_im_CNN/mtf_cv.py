from transformation import Transformation
from torchvision import transforms
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
from neptune.utils import stringify_unsupported
from joblib import Parallel, delayed
from multiprocessing import Manager
from sklearn.model_selection import KFold
from timeit import default_timer

import torch
import torch.nn as nn
import neptune
import matplotlib.pyplot as plt
import optuna
import numpy as np
import os
import pickle
import argparse
   
class CrossValidation:
    def __init__(self,PROJECT:str,API_TOKEN:str,transformation:Transformation,K,W_SIZE,MODEL_NAME,LOSS,device):
        #fixparameter
        self.k = K
        self.w_size = W_SIZE
        self.transformation = transformation
        self.device = device
        self.model_name = MODEL_NAME
        self.loss = LOSS
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
        self.load_dataset = self.transformation.load_dataset
        
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
    
    def transform_image (self, train_raw_data, test_raw_data, n_bins, strategy):
        #finction to transform raw_norm_data to image and normalize pixel it
        train_image = self.transformation.mtf(train_raw_data,n_bins=n_bins, strategy=strategy)
        test_image =  self.transformation.mtf(test_raw_data,n_bins=n_bins, strategy=strategy)
        train_image, test_image, scaler_pixel = self.transformation.normalize_pixel(train_image=train_image,test_image=test_image)
        return train_image, test_image, scaler_pixel  
    
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
 
    def load_scaler (self,split, test_raw_data:TensorDataset, n_bins, strategy, clamp = False):
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
        feature_test = torch.from_numpy(feature_test).float()
        
        #clamp
        if clamp  == True:
            feature_test = np.clip(feature_test,-1,1)
        
        #transform to TensorDataset
        test_data = TensorDataset(feature_test,label_test)
        
        #transform to image
        test_data_image = self.transformation.mtf(test_data, n_bins=n_bins, strategy=strategy)   
        feature_test, label_test = test_data_image.tensors
        
        file_path = os.path.join(self.dir,"scaler_pixel_cv.pkl")
        with open(file_path,"rb") as file:
            scaler_dict = pickle.load(file)
        scaler = scaler_dict[self.best_trial_number][split]
        
        feature_test = scaler(feature_test)    
        test_data_image = TensorDataset(feature_test,label_test)
        
        return self.load_dataset(test_data, test_data_image)
    
    def load_model (self,drop:float):
        #load concat model
        model = self.transformation.load_model(drop=drop,w_size = self.w_size)
        
        return model
    
    def data_loader(self,data,batch_size:int):
        loader = DataLoader(dataset=data,batch_size=batch_size,shuffle=True)
        return loader
    
    def score(self,y_pred,y_true):
        eps = torch.tensor(1e-7)
        Er = 100 * (y_true - y_pred) / torch.maximum(y_true,eps)
        A = torch.where(Er <= 0,torch.exp(-torch.log(torch.tensor(0.5)) * (Er / 5 )),torch.exp(torch.log(torch.tensor(0.5)) * (Er / 20 )))
        score = torch.mean(A)
        return score
        
    def train_evaluate_loop(self,run:neptune.init_run,model:nn.Module,train_loader,val_loader,epochs,optimizer:torch.optim,cv:bool=True,evaluate:bool = False):
        
        #check if evaluate
        if evaluate:
            epochs = 1
            model_path = os.path.join(self.dir,"t_{}_final.pth".format(self.best_trial_number))
            state_dict = torch.load(model_path,map_location=self.device)
            model.load_state_dict(state_dict)
            
        #training loop
        for iter in range(epochs):
            #score
            score_train = 0
            loss_train = 0
            
            score_val = 0
            loss_val = 0
            
            #tracking list rul
            if cv == True:    
                tracking_cv = {}
                tracking_cv.setdefault("train",[[],[]])
                tracking_cv.setdefault("val",[[],[]])
                
            else:
                tracking_train = {}
                for bearing in self.bearing_train_number:
                    tracking_train.setdefault(bearing,[[],[]])
                
                tracking_test = {}
                for bearing in self.bearing_test_number:
                    tracking_test.setdefault(bearing,[[],[]])
                    
            for X_train_ts,X_train_im,y_train in train_loader:
                
                #training mode
                if evaluate:
                    model.eval()
                else:    
                    model.train()
                
                #bearing from tensor
                if cv == False:
                    bearing_train = y_train[:,1].int()
                    
                #load data to device
                X_train_ts = X_train_ts.to(self.device)
                X_train_im = X_train_im.to(self.device)
                y_train = y_train[:,0].to(self.device)
                
                #forward pass
                y_train_pred = model(X_train_ts,X_train_im).ravel()
                
                #turn off requires grad if evaluate mode
                if evaluate:
                    y_train_pred = y_train_pred.detach()
                    
                #calculate the loss
                loss_train_this_batch = self.loss(y_train_pred,y_train)
                loss_train = loss_train + loss_train_this_batch
                
                #clamp
                y_train_pred = y_train_pred.clamp(0,1)
                
                #calculate the score
                score_train_this_batch = self.score(y_train_pred,y_train)
                score_train = score_train + score_train_this_batch
                
                #check evaluate to applied gradient decent
                if not evaluate:
                    
                    #gradient zero grad
                    optimizer.zero_grad()
                    
                    #backpropagation
                    loss_train_this_batch.backward()
                    
                    #update the weights and bias
                    optimizer.step()
                
                #indexing tracking
                if cv == True:
                    tracking_cv["train"][0] = tracking_cv["train"][0] + y_train_pred.tolist()
                    tracking_cv["train"][1] = tracking_cv["train"][1] + y_train.tolist()
                    
                else:    
                    for bear in self.bearing_train_number:
                        tracking_train[bear][0] = tracking_train[bear][0] + y_train_pred[bearing_train==bear].tolist()
                        tracking_train[bear][1] = tracking_train[bear][1] + y_train[bearing_train==bear].tolist()
                        
            #evaluating
            model.eval()    
            with torch.inference_mode():
                for X_val_ts,X_val_im,y_val in val_loader:
                    
                    #load bering
                    if cv == False:
                        bearing_test = y_val[:,1].int()
                    
                    #load data to device
                    X_val_ts = X_val_ts.to(self.device)
                    X_val_im = X_val_im.to(self.device)
                    y_val = y_val[:,0].to(self.device)
                    
                    #forward pass
                    y_val_pred = model(X_val_ts,X_val_im).ravel()
                    
                    #calculate the loss
                    loss_val_this_batch = self.loss(y_val_pred,y_val)
                    loss_val = loss_val + loss_val_this_batch
                    
                    #clamp
                    y_val_pred = y_val_pred.clamp(0,1)
                    
                    #calculate the score
                    score_val_this_batch = self.score(y_val_pred,y_val)
                    score_val = score_val + score_val_this_batch
                    
                    #indexing tracking
                    if cv == True: 
                        tracking_cv["val"][0] = tracking_cv["val"][0] + y_val_pred.tolist()
                        tracking_cv["val"][1] = tracking_cv["val"][1] + y_val.tolist()
                    else:
                        for bear in self.bearing_test_number:
                            tracking_test[bear][0] = tracking_test[bear][0] + y_val_pred[bearing_test==bear].tolist()
                            tracking_test[bear][1] = tracking_test[bear][1] + y_val[bearing_test==bear].tolist()
                        
            #mean loss/score
            loss_train =  loss_train.item() / len(train_loader)
            loss_val =  loss_val.item() / len(val_loader)
            
            score_train =  score_train.item() / len(train_loader)
            score_val =  score_val.item() / len(val_loader) 
            
            #log image and metrics
            if cv == True:
                fig_rul = self.plot_rul(kind_data="cv",iter=iter,tracking_dict=tracking_cv)
                run["images/RUL train val"].append(fig_rul,step=iter)
                plt.close()
                metric = {"score train":score_train,"score val":score_val,"loss train":loss_train,"loss val":loss_val} 
                
                #check pruned:
                if N_JOBS_CV == 1:
                    self.trial.report(loss_val,iter + self.split*epochs) 
                    if self.trial.should_prune():
                        run["status"] = "pruned"
                        run.stop()
                        raise optuna.exceptions.TrialPruned()
            else:
                fig_rul_train = self.plot_rul(kind_data="train",iter=iter,tracking_dict=tracking_train)
                run["images/train"].append(fig_rul_train,step=iter)
                plt.close()
                fig_rul_test, score_final = self.plot_rul(kind_data="test",iter=iter,tracking_dict=tracking_test)
                run["images/test"].append(fig_rul_test,step=iter)
                plt.close()
                metric = {"score train":score_train,"score test":score_val,"loss train":loss_train,"loss test":loss_val, "score final":score_final} 
                print("epoch {}, loss train {:.2f}, loss test {:.2f}, score train {:.2f}, score test {:.2f}, score final {:.2f}".format(iter,loss_train,loss_val,score_train,score_val,score_final))
            
            #log metrics  
            run["metrics"].append(metric,step=iter)
            
        #saved model:                   
        if cv == True:
            model_path = os.path.join(self.dir,"t_{}_s_{}.pth".format(self.trial.number,self.split)) 
            torch.save(model.state_dict(),model_path)
            
            return loss_train, score_train, loss_val, score_val
        
        else:
            if not evaluate:
                #delete the bad trials mdoel
                model_bad = sorted([file for file in os.listdir(self.dir) if file.startswith("t_")])
                for file in model_bad:
                    if not file.startswith("t_{}_s_".format(self.best_trial_number)):
                        os.remove(os.path.join(self.dir,file))
                        
                #save the best trained model
                model_path = os.path.join(self.dir,"t_{}_final.pth".format(self.best_trial_number))
                torch.save(model.state_dict(),model_path) 
            
            return loss_train, score_train, loss_val, score_val, score_final
        
    def cv_load_data_parallel (self,batch_size,n_bins:int, strategy:str,params_idx):
        
        #train index and bearing train, val
        split, (train_idx,val_idx) = params_idx
        
        #create train_data, val_data and normalize it
        train_raw_data, val_raw_data, scaler_raw = self.normalize_raw(train_val_raw_data=self.train_val_raw_data,train_index=train_idx,val_index=val_idx,min_max_scaler=MIN_MAX_SCALER,clamp=CLAMP)
        
        #transform raw data to image
        train_data, val_data, scaler_pixel = self.transform_image(train_raw_data=train_raw_data,test_raw_data=val_raw_data,n_bins=n_bins,strategy=strategy)          
        
        #create dataset
        train_data = self.load_dataset(train_raw_data,train_data)
        val_data = self.load_dataset(val_raw_data,val_data)
        
        #turn data to data loader
        train_loader = self.data_loader(data=train_data,batch_size=batch_size)   
        val_loader = self.data_loader(data=val_data,batch_size=batch_size)
        
        #save the scaler:
        self.save_scaler(split=split,scaler_raw = scaler_raw, scaler_pixel = scaler_pixel)
        
        return (train_loader, val_loader) 
    
    def cv_train_eval_parallel (self,drop:float,optimizer_name:str,lr:float,epochs:int,weight_decay:float, params_load):
        
        #params from load data parallel
        split, (train_loader, val_loader) = params_load
        self.split = split
        
        #load mode
        model = self.load_model(drop=drop)
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
        run["hyperparameters"] = hyperparameters
        run["fix_parameters"] =  stringify_unsupported(fix_parameters)      
        run["fix_interval"] = stringify_unsupported(fix_interval) 
        
        #training loop
        loss_train_this_split, score_train_this_split,loss_val_this_split,score_val_this_split = self.train_evaluate_loop(run=run,model=model,train_loader=train_loader,val_loader=val_loader,epochs=epochs,optimizer=optimizer)
        
        #tracking split score
        print("Split {}, loss train {:.2f}, score train {:.2f}, loss val {:.2f} score val {:.2f}".format(split,loss_train_this_split,score_train_this_split,loss_val_this_split,score_val_this_split))    
    
        run["status"] = "finished"
        run["runing_time"] = run["sys/running_time"]
        run.stop()
        
        return loss_train_this_split, score_train_this_split,loss_val_this_split,score_val_this_split
             
    def cross_validation(self,trial:optuna.trial.Trial,drop:float,n_bins:int, strategy:str,optimizer_name:str,lr:float,batch_size:int,epochs:int,weight_decay:float):
        
        #trial variable
        self.trial = trial
        self.trial_number = trial.number
        
        #kfold
        kf = KFold(n_splits=self.k,shuffle=True,random_state=SEED)
         
        #params from load data 
        load_data = Parallel(n_jobs=N_JOBS_LOAD_DATA)(delayed(self.cv_load_data_parallel)(batch_size,n_bins, strategy,params_idx) for params_idx in enumerate(kf.split(self.train_val_raw_data)))
        
        
        #params from cross validation train and eval 
        result = Parallel(n_jobs = N_JOBS_CV)(delayed(self.cv_train_eval_parallel)(drop,optimizer_name,lr, epochs, weight_decay, params_load) for params_load in enumerate(load_data))
        result = np.array(result)
        loss_train, score_train, loss_val, score_val =  result[:,0].mean(),result[:,1].mean(), result[:,2].mean(),result[:,3].mean()
    
        #tracking split score
        print("trial {}, loss train mean {:.2f}, score train mean {:.2f}, loss val mean {:.2f}, score val mean {:.2f}\n\n".format(trial.number,loss_train,score_train,loss_val, score_val))  
    
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
            for X_test_ts,X_test_im,y_test in test_loader:
            
                #load bering
                bearing_test = y_test[:,1].int()
                
                #load data to device
                X_test_ts = X_test_ts.to(self.device)
                X_test_im = X_test_im.to(self.device)
                y_test = y_test[:,0].to(self.device)
                
                #forward pass
                y_test_pred = model(X_test_ts,X_test_im).ravel()
                y_test_pred = y_test_pred.clamp(0,1)

                #indexing tracking
                for bear in self.bearing_test_number:
                    tracking_test[bear][0] = tracking_test[bear][0] + y_test_pred[bearing_test==bear].tolist()
                    tracking_test[bear][1] = tracking_test[bear][1] + y_test[bearing_test==bear].tolist()

        #sort the tracking dict
        tracking_test = self.sort_trackinkg_cv(tracking_dict=tracking_test)
        return tracking_test
    
    def best_cv_final (self, run:neptune.init_run,model:torch.nn ,best_trial_number:int,full_raw_data:TensorDataset,batch_size, n_bins, strategy):
        
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
        test_data = Parallel(n_jobs=N_JOBS_LOAD_DATA_BEST_CV)(delayed(self.load_scaler)(split=split,test_raw_data=full_raw_data,n_bins=n_bins, strategy=strategy, clamp= CLAMP) for split in range(0,self.k))
        
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
        text = "\n\nTracking result from best trial in Cross Validation\n"
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
    
    def plot_rul(self,kind_data , tracking_dict,iter = None):
        
        if kind_data == "train" or kind_data == "test":
            #create figure and axes
            if kind_data == "train":
                fig, ax_fig = plt.subplots(figsize = (17,13),nrows=2,ncols=3)
                bearing_list = self.bearing_train_number
                fig.suptitle("RUL of bearing train epoch {}".format(iter))
                    
            elif kind_data == "test":
                fig,ax_fig = plt.subplots(figsize = (20,15),nrows=3,ncols=4)
                bearing_list = self.bearing_test_number
                if iter == None:
                    fig.suptitle("RUL of bearing test best CV")
                else:
                    fig.suptitle("RUL of bearing test epoch {}".format(iter))
                rul_last_predict = np.empty(shape=(len(bearing_list,)))
                rul_last_true = np.empty(shape=(len(bearing_list,)))
                loss_final = np.empty(shape=(len(bearing_list,)))
            
            axes = [item for sublist in ax_fig for item in sublist]
                
            for index, bearing in enumerate(bearing_list):
                ax = axes[index]
                #sort the index of the rul true
                sort_index = np.argsort(tracking_dict[bearing][1]) #low to high
                time_stamp = np.array(tracking_dict[bearing][1])[sort_index]
                rul_true = np.array(tracking_dict[bearing][1])[np.flip(sort_index)] #high to low
                rul_pred = np.array(tracking_dict[bearing][0])[np.flip(sort_index)]

                if kind_data == "test":
                    rul_last_true[index] = rul_true[self.lc_window_index[bearing]]
                    rul_last_predict[index] = rul_pred[self.lc_window_index[bearing]]
                    score_final = self.score(y_pred=torch.from_numpy(np.array(rul_pred[self.lc_window_index[bearing]])),
                                            y_true=torch.from_numpy(np.array(rul_true[self.lc_window_index[bearing]])))
                    
                #calculate score and loss
                loss = self.loss(torch.from_numpy(rul_pred),torch.from_numpy(rul_true))
                score = self.score(y_pred=torch.from_numpy(rul_pred),y_true=torch.from_numpy(rul_true))
                
                ax.plot(time_stamp,rul_pred)
                ax.plot(time_stamp,rul_true)
                if kind_data == "test":
                    ax.scatter(time_stamp[self.lc_window_index[bearing]],rul_pred[self.lc_window_index[bearing]],s = 30)
                    ax.scatter(time_stamp[self.lc_window_index[bearing]],rul_true[self.lc_window_index[bearing]],s = 30)
                ax.set_xlabel("time stamp")
                ax.set_ylabel("RUL")
                ax.legend(["RUL predict","RUL true"],loc="upper right")
                
                if kind_data == "train":
                    ax.set_title("Bearing {}, loss {:.2f}, score {:.2f}".format(bearing,loss,score))
                elif kind_data == "test":
                    rul_true_bearing = self.inverse_transform(bearing=bearing,label=rul_true[self.lc_window_index[bearing]])
                    rul_pred_bearing = self.inverse_transform(bearing=bearing,label=rul_pred[self.lc_window_index[bearing]])
                    loss_final_bearing = self.loss(torch.tensor(rul_pred_bearing),torch.tensor(rul_true_bearing)).item()
                    loss_final[index] = loss_final_bearing
                    ax.set_title("Bearing {}, loss {:.2f}, score {:.2f}, loss final {}\nscore final {:.2f}, RUL true {}, RUL pred {}".format(bearing,loss,score,round(loss_final_bearing),score_final,round(rul_true_bearing),round(rul_pred_bearing)))
            
            if kind_data == "test":
                loss_final = loss_final.mean()
                score_final = self.score(y_pred=torch.from_numpy(rul_last_predict),y_true=torch.from_numpy(rul_last_true)).item()
                
                ax = axes[-1]
                ax.axis("off")   
                ax.text(0.2,0.5,"loss final = {}\nscore final = {:.2f}".format(round(loss_final),score_final),fontsize = 15)

                fig.subplots_adjust(left=0.05,bottom=0.079,right=0.95,top=0.917,wspace=0.224,hspace=0.317)
                
                return fig,score_final
                    
            elif kind_data == "train":
                return fig  

        else:
            #create figure
            fig, axes = plt.subplots(ncols=2,nrows=1,figsize = (17,5))
            fig.suptitle("tracking RUL train/val split {} epoch {}".format(self.split,iter))
            
            for index, kind in enumerate(["train","val"]):
                
                #sort the index of rul train
                samples = np.arange(0,len(tracking_dict[kind][0]))
                sorted_index = np.argsort(tracking_dict[kind][1]) #low to high
                
                rul_true = np.array(tracking_dict[kind][1])[np.flip(sorted_index)]
                rul_pred = np.array(tracking_dict[kind][0])[np.flip(sorted_index)]
                
                #calculate score and loss
                loss = self.loss(torch.from_numpy(rul_pred),torch.from_numpy(rul_true))
                score = self.score(y_pred=torch.from_numpy(rul_pred),y_true=torch.from_numpy(rul_true))
            
                #plot RUL
                ax = axes[index]
                ax.plot(samples,rul_pred)
                ax.plot(samples,rul_true)
                ax.set_xlabel("n_samples")
                ax.set_ylabel("RUL scaled")
                ax.legend(["RUL predict","RUL true"])
                ax.set_title("RUL {}, loss {:.2f}, score {:.2f}".format(kind, loss, score))
                
            return fig
        
    def rename_best_cv_file(self, best_trial_number):
        for file in sorted(os.listdir(self.dir)):
            if (file.startswith("t_{}_".format(best_trial_number))) and ("final" not in file) and ("best" not in file):
                old_name = os.path.join(self.dir,file)
                new_name = os.path.join(self.dir,file.replace(".pth","_best.pth"))
                os.rename(old_name,new_name)
                                        
    def database_to_csv(self,name_method:str, storage_name:str):
        #turn database to csv file
        study = optuna.load_study(study_name=name_method,storage=storage_name)   
        df = study.trials_dataframe()
        name_file = os.path.join(self.dir,"trials_data.csv")
        df.to_csv(name_file, index=False)             
                       
    def final_test (self,best_params:dict,best_trial_number, evaluate = False):
        
        #print the test
        if evaluate == False:
            print("\nTrain model with best hyperparameters on train data and evaluate on test data.\n")
        else:
            print("\nEvaluate model with best hyperparameters on train data and on test data.\n")
        start = default_timer()
          
        #load hyperparamters
        n_bins, strategy = best_params["n_bins"], best_params["strategy"]
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
        train_raw_data, full_raw_data, _ = self.normalize_raw(train_val_raw_data=self.train_val_raw_data,test_raw_data=full_data_raw,min_max_scaler=MIN_MAX_SCALER, clamp=CLAMP)
        
        #transform data to images
        train_data, full_data, _ = self.transform_image(train_raw_data=train_raw_data,test_raw_data=full_raw_data,n_bins=n_bins, strategy=strategy)
        fig_transform = self.visualize_transform_image(train_val_data=train_data)
        
        #print the time for data preprocessing
        end = default_timer()
        print("Datapreprocessing takes {} seconds".format(end-start))
        
        #create dataset
        train_data = self.load_dataset(train_raw_data,train_data)
        full_data = self.load_dataset(full_raw_data,full_data)
        
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
        
        run["images/transform"] = fig_transform
        plt.close()
        
        #optimizer and loss
        optimizer = getattr(torch.optim, optimizer)(params = model.parameters(), lr = lr )
        
        #training, testing model
        start = default_timer()
        loss_train, score_train, loss_test, score_test,  score_final = self.train_evaluate_loop(run=run,model=model,train_loader=train_loader,val_loader=full_loader,epochs=epochs,optimizer=optimizer,cv=False,evaluate=evaluate)   
        
        #print the time for training and testing 
        end = default_timer()
        print("Training/testing takes {} seconds".format(end-start))
        
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
        run["summary"] = model_summary
        model_structure = model.__repr__()
        run["structure"] = model_structure
        
        #final best 6 cv
        end = default_timer()
        text = self.best_cv_final(run=run,model=model,best_trial_number=best_trial_number,full_raw_data=full_data_raw,batch_size=batch_size,n_bins=n_bins, strategy=strategy)
        print(text)

        #print the time for training and testing 
        end = default_timer()
        print("Training/testing takes {} seconds".format(end-start))
        
        #finish
        run["status"] = "finished"
        run["runing_time"] = run["sys/running_time"]
        run.stop()
        
        #rename best cv file
        self.rename_best_cv_file(best_trial_number=best_trial_number)
        
        return loss_train, score_train, loss_test, score_test, score_final        
                               
#creat objective trial optuna
def objective(trial:optuna.trial.Trial):
    
    #hyperparameter method
    n_bins=trial.suggest_int(name="n_bins",low=N_BINS[0],high=N_BINS[1],step=N_BINS[2]) 
    strategy = trial.suggest_categorical(name="strategy",choices=STRATEGY)
    
    #hyperparameter model
    drop = trial.suggest_float(name="drop",low=DROP[0],high=DROP[1],log=DROP[2])
    optimizer_name = trial.suggest_categorical(name="optimizer",choices=OPTIMIZER)
    lr = trial.suggest_float(name="lr",low=LR[0],high=LR[1],log=LR[2])
    weight_decay = trial.suggest_float(name="weight_decay",low=WEIGHT_DECAY[0],high=WEIGHT_DECAY[1],log=WEIGHT_DECAY[2])
    
    #fix hyperparameter
    batch_size = trial.suggest_categorical(name="batch_size",choices=BATCH_SIZE)
    epochs = trial.suggest_categorical(name="epochs",choices=EPOCHS)
    
    #cross validation
    loss_val = cross.cross_validation(trial=trial,drop=drop,n_bins = n_bins,strategy = strategy,optimizer_name=optimizer_name,lr=lr,batch_size=batch_size,epochs=epochs,weight_decay=weight_decay)
    
    return loss_val

#project 
PROJECT = "ba-final/mtf-2"
API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ=="
METHOD = "MTF"

N_JOBS_LOAD_DATA = -1
N_JOBS_CV = 1
N_JOBS_LOAD_DATA_BEST_CV = 1
N_JOBS_BEST_CV = 1

MIN_MAX_SCALER = True
CLAMP = False

#Fixparameters               
SEED = 1998
INPUT = 2
OUTPUT = 1
K = 5
TOTAL_TRIALS = 100
W_SIZE = 25600
MODEL_NAME = "efficientnet_b0"
BATCH_SIZE = [32]
EPOCHS = [100]
n_trials = 1

fix_parameters = {"k":K,"seed":SEED,"model_name":MODEL_NAME,"w_size":W_SIZE,"n_trials":TOTAL_TRIALS,"batch_size":BATCH_SIZE,"epochs":EPOCHS}

#interval 
N_BINS = (5,40,1)
STRATEGY = ("uniform","quantile","normal")

DROP = (0.1,0.8,True)
OPTIMIZER = ("Adam","SGD")
LR = (0.0005,0.2,True)
WEIGHT_DECAY = (0.00001,0.1,True)
fix_interval = {"optmizer":OPTIMIZER,"learning_rata":LR,"weight_decay":WEIGHT_DECAY,"n_bins":N_BINS,"strategy":STRATEGY}

#check device
torch.manual_seed(SEED)
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
loss = nn.L1Loss()

#argparse
parse = argparse.ArgumentParser(description="this file is using method {} as preprocessing and try to find the best hyperparameter by using cross validation on train and validation data. Finaly train the model on train and validation data with best hyperparameters and evaluate on test data. To run this file, please use python3 {}_cv.py -d dir_name -m run_mode. More details about the arguments can be seen below. ".format(METHOD, METHOD.lower()))
parse.add_argument("-d","--dir_name",type=str, default="model_pretrained",
                   help='name of the directory store of pretrained models, database file of the trials and the scalers. Default is "model_pretrained".')
parse.add_argument("-m","--run_mode",type=str, default="optimize", choices=["optimize","evaluate"],
                   help='choose a mode to run.\n "optimize" is trying to optimize the hyperparameter with 100 trials by using cross validation, if already a database file to store the trials exsist, try to run until 100 trials and choose the best trial with best hyperparameter to train a new model on train data and evaluate on test data, else create a new database file and run from trial 0. "evaluate" is trying to evaluate a pretrained model on test data. If a pretrained model is not available, "optimize" is choosen. Default is "optimize".')

args = parse.parse_args()

#cross validation
cross = CrossValidation(PROJECT=PROJECT,API_TOKEN=API_TOKEN,transformation=transformation,K = K,W_SIZE=W_SIZE,MODEL_NAME=MODEL_NAME,LOSS=loss,device=device)
dir_name = args.dir_name
dir_name = cross.dir = "{}/{}".format(dir_name,METHOD.lower())
os.makedirs(dir_name,exist_ok=True)
print("The pretrained models are saved at {}.\n\n".format(dir_name))

#create study
storage_path = os.path.join(dir_name,"{}.db".format(METHOD))
storage_name = "sqlite:///{}".format(storage_path)

if not os.path.isfile(storage_path):
    study = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=SEED),study_name=METHOD,storage=storage_name,pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5, interval_steps=1, n_min_trials=9))
    study.optimize(objective,n_trials=n_trials)    
    print('\nRun mode: "optimize".\n')
    
else:
    study = optuna.load_study(study_name=METHOD, storage=storage_name,sampler=optuna.samplers.TPESampler(seed = SEED),pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5, interval_steps=1, n_min_trials=9))
    
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
        print('\nRun mode: "optimize".\n')
        
    else: 
        #find the best hyperparameters
        best_trial_score = study.best_trial
        best_trial_params = best_trial_score.params
        
        #check argument and model path 
        run_mode = args.run_mode
        model_path = os.path.join(dir_name,"t_{}_final.pth".format(best_trial_score.number))
        evaluate = True if (run_mode == "evaluate" and os.path.exists(model_path)) else False
        run_mode = "evaluate" if evaluate == True else "optimize"
        print('\nRun mode: {}.\n'.format(run_mode))
        
        print("Study statistics: ")
        print("  Number of finished trials: ", pruned_trials + complete_trials)
        print("  Number of pruned trials: ", pruned_trials)
        print("  Number of complete trials: ", complete_trials)

        print("  Value: ", best_trial_score.value)
        print("  Trial number: ", best_trial_score.number)
        print("  Params: ")
        for key, value in best_trial_params.items():
            print("    {}: {}".format(key, value))

        loss_train, score_train, loss_test, score_test, score_final = cross.final_test(best_trial_params,best_trial_score.number,evaluate) #22best_trial_score.number

        #merge all the result text file
        transformation.merge_result(METHOD)
        
        #turn database to csv file
        cross.database_to_csv(name_method=METHOD,storage_name=storage_name)