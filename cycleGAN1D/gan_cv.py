from transformation import Transformation
from torchvision import transforms
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
from neptune.utils import stringify_unsupported
from joblib import Parallel, delayed
from multiprocessing import Manager
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import neptune
import matplotlib.pyplot as plt
import optuna
import numpy as np
import os
import pickle
    
class CrossValidation:
    def __init__(self,PROJECT:str,API_TOKEN:str,TRANSOFRMATION:Transformation,K,W_SIZE,HIDDEN_CHANNELS_GEN,HIDDEN_CHANNELS_DIS,MAE,MSE,DEVICE):
        #fixparameter
        self.k = K
        self.w_size = W_SIZE
        self.hidden_channels_gen = HIDDEN_CHANNELS_GEN
        self.hidden_cahnnels_dis = HIDDEN_CHANNELS_DIS
        self.transformation = TRANSOFRMATION
        self.device = DEVICE
        self.mae = MAE
        self.mse = MSE
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
        self.load_dataset = self.transformation.load_datasets
        
        #saved variable:
        self.source_raw_data = self.load_raw_data(kind_data="train")
        self.target_raw_data = self.load_raw_data(kind_data="full")
        self.target_split_index_cv = self.target_split()
        self.scaler_raw_cv = Manager().dict() 
        self.scaler_pixel_cv = Manager().dict()
        self.scaler_pixel_total = Manager().dict() 
        self.dir = "model_pretrained/{}".format(self.project.split("/")[1].split("-")[0]) 
    
    def load_raw_data(self,kind_data:str):
        #function to load raw data
        raw_data = self.transformation.load_raw_data(kind_data=kind_data,window_size=self.w_size)
        return raw_data
    
    def save_scaler (self,split, scaler_raw):
        
        #function to check if saved scaler available in directory model_saved
        file_path_scaled_raw_cv = os.path.join(self.dir,"scaler_raw_cv.pkl")
                
        if not os.path.isfile(file_path_scaled_raw_cv):
            if len(self.scaler_raw_cv) < self.k: 
                self.scaler_raw_cv[split]=scaler_raw 
                
            if len(self.scaler_raw_cv) == self.k: 
                with open(file_path_scaled_raw_cv,"wb") as file:
                    pickle.dump(dict(self.scaler_raw_cv),file)

    def load_scaler (self,split, raw_data:TensorDataset, clamp = False):
        #function to load a saved scaler and transform it to test_raw_data
        file_path = os.path.join(self.dir,"scaler_raw_cv.pkl")
        
        with open(file_path,"rb") as file:
            scaler_dict = pickle.load(file)
        
        #split feature, label and reshape
        feature_test, label_test = raw_data.tensors
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
        test_data = self.transformation.lms(test_data,w_stft,hop,n_mels)  
        feature_test, label_test = test_data.tensors
        
        file_path = os.path.join(self.dir,"scaler_pixel_cv.pkl")
        with open(file_path,"rb") as file:
            scaler_dict = pickle.load(file)
        scaler = scaler_dict[self.best_trial_number][split]
        
        feature_test = scaler(feature_test)    
        
        return TensorDataset(feature_test,label_test)
    
    #function to get index of train val source index consider on bearing test in CV.
    def target_split (self):        
        return self.transformation.target_split(w_size=self.w_size,k =self.k)
    
    def load_model (self):
        gen, dis = self.transformation.load_model(in_channels=2, hidden_channels_gen = self.hidden_channels_gen, hidden_channels_dis = self.hidden_cahnnels_dis)
        return gen,dis 
    
    def data_loader(self,data,batch_size:int):
        loader = DataLoader(dataset=data,batch_size=batch_size,shuffle=True)
        return loader
    
    def score(self,y_pred,y_true):
        eps = torch.tensor(1e-7)
        Er = 100 * (y_true - y_pred) / torch.maximum(y_true,eps)
        A = torch.where(Er <= 0,torch.exp(-torch.log(torch.tensor(0.5)) * (Er / 5 )),torch.exp(torch.log(torch.tensor(0.5)) * (Er / 20 )))
        score = torch.mean(A)
        return score
        
    def train_evaluate_loop(self,run:neptune.init_run,gen_S:nn.Module, gen_T:nn.Module,dis_S:nn.Module,dis_T:nn.Module,train_loader,val_loader,epochs,opt_d:torch.optim,opt_g:torch.optim,lambda_cyc, lambda_idt,cv:bool=True):
 
        #training loop
        for iter in range(epochs):
            #all losses train
            loss_d_train = 0
            loss_g_train = 0
            loss_cyc_train = 0
            loss_idt_train = 0
            loss_train_raw = 0
            
            #all losses val
            loss_d_val = 0
            loss_g_val = 0
            loss_cyc_val = 0
            loss_idt_val = 0
            loss_val_raw = 0
            
            """#tracking list rul
            tracking_train = {}
            for bearing in self.bearing_train_number:
                tracking_train.setdefault(bearing,[[],[]])
            
            tracking_test = {}
            for bearing in self.bearing_test_number:
                tracking_test.setdefault(bearing,[[],[]])"""
        
            #training mode
            gen_S.train()       
            gen_T.train()   
            dis_S.train()    
            dis_T.train()   
            
            for source_train,target_train, label_train in train_loader:
                
                #bearing from tensor
                bearing_train = target_train[:,1].int()
                label_train = label_train[:,0]  
                
                #load data to device
                source_train = source_train.float().to(self.device)
                target_train = target_train.float().to(self.device)
                
                #create fake source and target
                source_train_fake = gen_S(target_train)  #source ≈ Horse , target ≈ Zebra
                target_train_fake = gen_T(source_train)
                
                #train dis
                d_S_train_real = dis_S(source_train)
                d_S_train_fake = dis_S(source_train_fake.detach())
                
                d_T_train_real = dis_T(target_train)
                d_T_train_fake = dis_T(target_train_fake.detach())
                
                #calculate the loss dis
                loss_d_train_real = self.mse(d_S_train_real,torch.ones_like(d_S_train_real)) + self.mse(d_T_train_real,torch.ones_like(d_T_train_real))
                loss_d_train_fake = self.mse(d_S_train_fake,torch.zeros_like(d_S_train_fake)) + self.mse(d_T_train_fake,torch.zeros_like(d_T_train_fake))
                loss_d_train_this_batch = (loss_d_train_real + loss_d_train_fake) / 2
                loss_d_train = loss_d_train + loss_d_train_this_batch
                
                #gradient decent and backpropagation
                opt_d.zero_grad()
                loss_d_train_this_batch.backward()
                opt_d.step()
                
                #train generator
                d_S_train_fake = dis_S(source_train_fake)
                d_T_train_fake = dis_T(target_train_fake)
                
                #caculate the loss gen
                loss_g_S_train = self.mse(d_S_train_fake,torch.ones_like(d_S_train_fake))
                loss_g_T_train = self.mse(d_T_train_fake,torch.ones_like(d_T_train_fake))
                
                #cycle loss
                if lambda_cyc > 0:
                    source_train_fake_fake = gen_S(target_train_fake)
                    target_train_fake_fake = gen_T(source_train_fake)
                    
                    loss_cyc_S_train = self.mae(source_train_fake_fake,source_train)
                    loss_cyc_T_train= self.mae(target_train_fake_fake,target_train)
                else:
                    loss_cyc_S_train = 0
                    loss_cyc_T_train = 0
                    
                loss_cyc_train = loss_cyc_S_train + loss_cyc_T_train + loss_cyc_train     
                
                #identity loss
                if lambda_idt > 0:
                    source_train_same = gen_S(source_train)
                    target_train_same = gen_T(target_train)
                    
                    loss_idt_S_train = self.mae(source_train_same,source_train)
                    loss_idt_T_train = self.mae(target_train_same,target_train)
                    
                else:
                    loss_idt_S_train = 0
                    loss_idt_T_train = 0
                
                loss_idt_train = loss_idt_S_train + loss_idt_T_train + loss_idt_train
                
                #calculate the loss:
                loss_g_train_this_batch = loss_g_S_train + loss_g_T_train + lambda_cyc*(loss_cyc_S_train + loss_cyc_T_train) + lambda_idt*(loss_idt_S_train+ loss_idt_T_train)
                loss_g_train = loss_g_train_this_batch + loss_g_train
                
                #calculate the loss raw 
                loss_train_raw_this_batch = loss_d_train_fake**2 + loss_d_train_real**2 + loss_g_S_train**2 + loss_g_T_train**2 + loss_cyc_S_train**2 + loss_cyc_T_train**2 + loss_idt_S_train**2 + loss_idt_T_train**2
                loss_train_raw = loss_train_raw_this_batch + loss_train_raw
                
                #gradient decent and backpropagation
                opt_g.zero_grad()
                loss_g_train_this_batch.backward()
                opt_g.step()    
                   
                #take only the gen loss
                loss_g_train_this_batch = loss_g_S_train + loss_g_T_train 
                loss_g_train = loss_g_train_this_batch + loss_g_train
                
                """#indexing tracking
                if cv == True:
                    tracking_cv["train"][0] = tracking_cv["train"][0] + y_train_pred.tolist()
                    tracking_cv["train"][1] = tracking_cv["train"][1] + target_train.tolist()
                    
                else:    
                    for bear in self.bearing_train_number:
                        tracking_train[bear][0] = tracking_train[bear][0] + y_train_pred[bearing_train==bear].tolist()
                        tracking_train[bear][1] = tracking_train[bear][1] + target_train[bearing_train==bear].tolist()"""
                        
            #evaluating
            gen_S.eval()       
            gen_T.eval()   
            dis_S.eval()    
            dis_T.eval()    
            with torch.inference_mode():
                for source_val,target_val, label_val in val_loader:
                    
                    #bearing from tensor
                    bearing_val = target_val[:,1].int()
                    label_val = label_val[:,0]
                    
                    #load data to device
                    source_val = source_val.float().to(self.device)
                    target_val = target_val.float().to(self.device)
                    
                    #create fake source and target
                    source_val_fake = gen_S(target_val)  #source ≈ Horse , target ≈ Zebra
                    target_val_fake = gen_T(source_val)
                    
                    #calculate the loss dis
                    d_S_val_real = dis_S(source_val)
                    d_S_val_fake = dis_S(source_val_fake.detach())
                    
                    d_T_val_real = dis_T(target_val)
                    d_T_val_fake = dis_T(target_val_fake.detach())

                    loss_d_val_real = self.mse(d_S_val_real,torch.ones_like(d_S_val_real)) + self.mse(d_T_val_real,torch.ones_like(d_T_val_real))
                    loss_d_val_fake = self.mse(d_S_val_fake,torch.zeros_like(d_S_val_fake)) + self.mse(d_T_val_fake,torch.zeros_like(d_T_val_fake))
                    loss_d_val_this_batch = (loss_d_val_real + loss_d_val_fake) / 2
                    loss_d_val = loss_d_val + loss_d_val_this_batch
                    
                    #caculate the loss gen
                    loss_g_S_val = self.mse(d_S_val_fake,torch.ones_like(d_S_val_fake))
                    loss_g_T_val = self.mse(d_T_val_fake,torch.ones_like(d_T_val_fake))
                    
                    #cycle loss
                    if lambda_cyc > 0:
                        source_val_fake_fake = gen_S(target_val_fake)
                        target_val_fake_fake = gen_T(source_val_fake)
                        
                        loss_cyc_S_val = self.mae(source_val_fake_fake,source_val)
                        loss_cyc_T_val= self.mae(target_val_fake_fake,target_val)
                    else:
                        loss_cyc_S_val = 0
                        loss_cyc_T_val = 0
                        
                    loss_cyc_val = loss_cyc_S_val + loss_cyc_T_val + loss_cyc_val
                    
                    #identity loss
                    if lambda_idt > 0:
                        source_val_same = gen_S(source_val)
                        target_val_same = gen_T(target_val)
                        
                        loss_idt_S_val = self.mae(source_val_same,source_val)
                        loss_idt_T_val = self.mae(target_val_same,target_val)
                        
                    else:
                        loss_idt_S_val = 0
                        loss_idt_T_val = 0
                    
                    loss_idt_val = loss_idt_S_val + loss_idt_T_val + loss_idt_val
                    
                    #calculate the loss:
                    loss_g_val_this_batch = loss_g_S_val + loss_g_T_val 
                    loss_g_val = loss_g_val_this_batch + loss_g_val
                    
                    #calculate the loss raw 
                    loss_val_raw_this_batch = loss_d_val_fake**2 + loss_d_val_real**2 + loss_g_S_val**2 + loss_g_T_val**2 + loss_cyc_S_val**2 + loss_cyc_T_val**2 + loss_idt_S_val**2 + loss_idt_T_val**2
                    loss_val_raw = loss_val_raw_this_batch + loss_val_raw
                    
                    """#indexing tracking
                    if cv == True: 
                        tracking_cv["val"][0] = tracking_cv["val"][0] + y_val_pred.tolist()
                        tracking_cv["val"][1] = tracking_cv["val"][1] + y_val.tolist()
                    else:
                        for bear in self.bearing_test_number:
                            tracking_test[bear][0] = tracking_test[bear][0] + y_val_pred[bearing_test==bear].tolist()
                            tracking_test[bear][1] = tracking_test[bear][1] + y_val[bearing_test==bear].tolist()"""
                        
            #mean loss/score
            loss_d_train =  loss_d_train.item() / len(train_loader)
            loss_g_train =  loss_g_train.item() /len(train_loader)
            loss_cyc_train = loss_cyc_train.item() / len(train_loader)
            loss_idt_train = loss_idt_train.item() / len(train_loader)
            loss_train_raw = loss_train_raw.item() / len(train_loader)
            
            loss_d_val =  loss_d_val.item() / len(val_loader)
            loss_g_val =  loss_g_val.item() /len(val_loader)
            loss_cyc_val = loss_cyc_val.item() / len(val_loader)
            loss_idt_val = loss_idt_val.item() / len(val_loader)
            loss_val_raw = loss_val_raw.item() / len(val_loader)
            
            #log image and metrics
            if cv == True:
                """fig_rul = self.plot_rul(kind_data="cv",iter=iter,tracking_dict=tracking_cv)
                run["images/RUL train val"].append(fig_rul,step=iter)
                plt.close()
                metric = {"score train":score_train,"score val":score_val,"loss train":loss_d_train,"loss val":loss_d_val} 
                """
                #check pruned:
                if N_JOBS_CV == 1:
                    self.trial.report(loss_val_raw,iter + self.split*epochs) 
                    if self.trial.should_prune():
                        run["status"] = "pruned"
                        run.stop()
                        raise optuna.exceptions.TrialPruned()
            """else:
                fig_rul_train = self.plot_rul(kind_data="train",iter=iter,tracking_dict=tracking_train)
                run["images/train"].append(fig_rul_train,step=iter)
                plt.close()
                fig_rul_test, score_final = self.plot_rul(kind_data="test",iter=iter,tracking_dict=tracking_test)
                run["images/test"].append(fig_rul_test,step=iter)
                plt.close()
                metric = {"score train":score_train,"score test":score_val,"loss train":loss_d_train,"loss test":loss_d_val, "score final":score_final} 
                print("epoch {}, loss train {:.2f}, loss test {:.2f}, score train {:.2f}, score test {:.2f}, score final {:.2f}".format(iter,loss_d_train,loss_d_val,score_train,score_val,score_final))"""
            
            #log metrics  
            metrics = {"loss_d_train":loss_d_train, "loss_g_train": loss_g_train, "loss_cyc_train":loss_cyc_train,"loss_idt_train":loss_idt_train,"loss_train_raw":loss_train_raw,
                       "loss_d_val" :loss_d_val,"loss_g_val":loss_g_val ,"loss_cyc_val":loss_cyc_val,"loss_idt_val":loss_idt_val,"loss_val_raw":loss_val_raw}
            run["metrics"].append(metrics,step=iter)
            
        #saved model:                   
        if cv == True:
            """ model_path = os.path.join(self.dir,"t_{}_s_{}.pth".format(self.trial.number,self.split)) 
            torch.save(model.state_dict(),model_path)"""
            
            return loss_d_train,loss_g_train, loss_cyc_train, loss_idt_train, loss_train_raw, loss_d_val, loss_g_val, loss_cyc_val,loss_idt_val ,loss_val_raw 
        """ else:
            #delete the bad trials mdoel
            model_bad = sorted([file for file in os.listdir(self.dir) if file.startswith("t_")])
            for file in model_bad:
                if not file.startswith("t_{}_s_".format(self.best_trial_number)):
                    os.remove(os.path.join(self.dir,file))
                    
            #save the best trained model
            model_path = os.path.join(self.dir,"t_{}_final.pth".format(self.best_trial_number))
            torch.save(model.state_dict(),model_path) 
            return loss_d_train, score_train, loss_d_val, score_val, score_final"""
    
    def cv_load_data_parallel (self,batch_size,params_idx):
        
        #train index and bearing train, val
        split, (train_idx_source,val_idx_source) = params_idx
        train_idx_target, val_idx_target = self.target_split_index_cv[split]
        
        #create train_data, val_data and normalize it
        source_train_data, source_val_data, scaler_source = self.normalize_raw(train_val_raw_data=self.source_raw_data,train_index=train_idx_source,val_index=val_idx_source,min_max_scaler=MIN_MAX_SCALER,clamp=CLAMP)
        target_train_data, target_val_data, scaler_source = self.normalize_raw(train_val_raw_data=self.target_raw_data,train_index=train_idx_target,val_index=val_idx_target,min_max_scaler=MIN_MAX_SCALER,clamp=CLAMP,scaler=scaler_source)
        
        #create datesets of train source target and val source target
        train_data = self.load_dataset(source_train_data,target_train_data)
        val_data = self.load_dataset(source_val_data,target_val_data)
        
        #turn data to data loader
        train_loader = self.data_loader(data=train_data,batch_size=batch_size)   
        val_loader = self.data_loader(data=val_data,batch_size=batch_size)
        print("split shape in cv load DONE:", split)
        #save the scaler:
        self.save_scaler(split=split,scaler_raw = scaler_source)
        
        return (train_loader, val_loader)  
    
    def cv_train_eval_parallel (self,optimizer_name:str,lr_g,lr_d:float, betas_g, betas_d,epochs:int,weight_decay_g:float,weight_decay_d:float,lambda_cyc, lambda_idt, params_load):
        
        #params from load data parallel
        split, (train_loader, val_loader) = params_load
        self.split = split
        
        #load mode
        gen_S, dis_S = self.load_model() #source ≈ Horse , target ≈ Zebra
        gen_T, dis_T = self.load_model() #source ≈ Horse , target ≈ Zebra
        if torch.cuda.device_count() > 1:
            gen_S = nn.DataParallel(gen_S)
            dis_S = nn.DataParallel(dis_S)
            gen_T = nn.DataParallel(gen_T)
            dis_T = nn.DataParallel(dis_T)
        gen_S = gen_S.to(self.device)
        dis_S = dis_S.to(self.device)
        gen_T = gen_T.to(self.device)
        dis_T = dis_T.to(self.device)
        
        #optimizer
        opt_g = getattr(torch.optim,optimizer_name)(params = list(gen_S.parameters()) + list(gen_T.parameters()), lr = lr_g, weight_decay = weight_decay_g, betas = betas_d)
        opt_d = getattr(torch.optim,optimizer_name)(params = list(dis_S.parameters()) + list(dis_T.parameters()), lr = lr_d, weight_decay = weight_decay_d, betas = betas_g)
        
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
        loss_d_train,loss_g_train, loss_cyc_train, loss_idt_train, loss_train_raw, loss_d_val, loss_g_val, loss_cyc_val,loss_idt_val ,loss_val_raw = self.train_evaluate_loop(run=run,gen_S=gen_S,gen_T=gen_T,dis_S=dis_S, dis_T=dis_T,train_loader=train_loader,val_loader=val_loader,epochs=epochs,opt_g=opt_g,opt_d=opt_d,lambda_cyc=lambda_cyc,lambda_idt=lambda_idt)
        
        #tracking split score
        print("Split {}, loss d train {:.2f}, loss g train {:.2f}, loss cyc train {:.2f}, loss idt train {:.2f}, loss raw train {:.2f}, loss d val {:.2f}, loss g val {:.2f}, loss cyc val {:.2f}, loss idt val {:.2f}, loss raw val {:.2f},".format(split,loss_d_train,loss_g_train, loss_cyc_train, loss_idt_train, loss_train_raw, loss_d_val, loss_g_val, loss_cyc_val,loss_idt_val ,loss_val_raw))    
    
        run["status"] = "finished"
        run["runing_time"] = run["sys/running_time"]
        run.stop()
        return loss_d_train,loss_g_train, loss_cyc_train, loss_idt_train, loss_train_raw, loss_d_val, loss_g_val, loss_cyc_val,loss_idt_val ,loss_val_raw
             
    def cross_validation(self,trial:optuna.trial.Trial,optimizer_name:str,lr_g,lr_d:float, betas_g, betas_d,epochs:int,batch_size:int,weight_decay_g:float,weight_decay_d:float,lambda_cyc, lambda_idt):
        
        #trial variable
        self.trial = trial
        self.trial_number = trial.number
        
        #kfold
        kf = KFold(n_splits=self.k,shuffle=False)
         
        #params from load data 
        load_data = Parallel(n_jobs=N_JOBS_LOAD_DATA)(delayed(self.cv_load_data_parallel)(batch_size,params_idx) for params_idx in enumerate(kf.split(self.source_raw_data)))
        
        #params from cross validation train and eval 
        result = Parallel(n_jobs = N_JOBS_CV)(delayed(self.cv_train_eval_parallel)(optimizer_name,lr_g,lr_d,betas_g,betas_d, epochs, weight_decay_g, weight_decay_d, lambda_cyc,lambda_idt, params_load) for params_load in enumerate(load_data))
        result = np.array(result)
        loss_d_train,loss_g_train, loss_cyc_train, loss_idt_train, loss_train_raw, loss_d_val, loss_g_val, loss_cyc_val,loss_idt_val ,loss_val_raw =  result[:,0].mean(),result[:,1].mean(), result[:,2].mean(),result[:,3].mean(), result[:,4].mean(),result[:,5].mean(), result[:,6].mean(),result[:,7].mean(),result[:,8].mean(),result[:,9].mean()

        #tracking split score
        print("Trial {}, loss d train {:.2f}, loss g train {:.2f}, loss cyc train {:.2f}, loss idt train {:.2f}, loss raw train {:.2f}, loss d val {:.2f}, loss g val {:.2f}, loss cyc val {:.2f}, loss idt val {:.2f}, loss raw val {:.2f},".format(trial.number,loss_d_train,loss_g_train, loss_cyc_train, loss_idt_train, loss_train_raw, loss_d_val, loss_g_val, loss_cyc_val,loss_idt_val ,loss_val_raw))    
    
        return loss_val_raw
      
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
    
    def best_cv_final (self, run:neptune.init_run,model:torch.nn ,best_trial_number:int,full_raw_data:TensorDataset,batch_size,w_stft,hop,n_mels):
        
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
        test_data = Parallel(n_jobs=N_JOBS_LOAD_DATA_BEST_CV)(delayed(self.load_scaler)(split=split,test_raw_data=full_raw_data,w_stft=w_stft,hop=hop,n_mels=n_mels,clamp=CLAMP) for split in range(0,self.k))
        
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

    def final_test (self,best_params:dict,best_trial_number):
        
        #load hyperparamters
        w_stft, hop, n_mels = best_params["w_stft"], best_params["hop"], best_params["n_mels"]
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
        train_data, full_data, _ = self.normalize_raw(train_val_raw_data=self.source_raw_data,test_raw_data=full_data_raw,min_max_scaler=MIN_MAX_SCALER,clamp=CLAMP)
        
        #transform data to images
        train_data, full_data, _ = self.transform_image(train_raw_data=train_data,test_raw_data=full_data,w_stft=w_stft,hop=hop,n_mels=n_mels)
        
        #turn data to dataloader
        train_loader = self.data_loader(data=train_data,batch_size=batch_size)
        full_loader = self.data_loader(data=full_data,batch_size=batch_size)
        
        #create run
        run = neptune.init_run(project=self.project, api_token=self.api_otken)
        run_name = "trial_best_no_{}".format(best_trial_number)
        run["name"] = run_name
        
        #log parameters and transform images
        run["hyperparameters"] = best_params
        run["fix_parameters"] = fix_parameters      
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
        loss_train, score_train, loss_test, score_test,  score_final = self.train_evaluate_loop(run=run,model=model,train_loader=train_loader,val_loader=full_loader,epochs=epochs,opt_d=optimizer,cv=False)   
        
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
        text = self.best_cv_final(run=run,model=model,best_trial_number=best_trial_number,full_raw_data=full_data_raw,batch_size=batch_size,w_stft=w_stft,hop=hop,n_mels = n_mels)
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
    
    #hyperparameter model
    optimizer_name = trial.suggest_categorical(name="optimizer",choices=OPTIMIZER)
    lr_g = trial.suggest_float(name="lr_g",low=LR_G[0],high=LR_G[1],log=LR_G[2])
    lr_d = trial.suggest_float(name="lr_d",low=LR_D[0],high=LR_D[1],log=LR_D[2])
    weight_decay_g = trial.suggest_float(name="weight_decay_g",low=WEIGHT_DECAY_G[0],high=WEIGHT_DECAY_G[1],log=WEIGHT_DECAY_G[2])
    weight_decay_d = trial.suggest_float(name="weight_decay_d",low=WEIGHT_DECAY_D[0],high=WEIGHT_DECAY_D[1],log=WEIGHT_DECAY_D[2])
    betas_1_d = trial.suggest_float(name="betas_1_d",low=BETAS_1_D[0],high=BETAS_1_D[1],step=BETAS_1_D[2])
    betas_2_d = trial.suggest_float(name="betas_2_d",low=BETAS_2_D[0],high=BETAS_2_D[1],step=BETAS_2_D[2])
    betas_d = (betas_1_d,betas_2_d)
    betas_1_g = trial.suggest_float(name="betas_1_g",low=BETAS_1_G[0],high=BETAS_1_G[1],step=BETAS_1_G[2])
    betas_2_g = trial.suggest_float(name="betas_2_g",low=BETAS_2_G[0],high=BETAS_2_G[1],step=BETAS_2_G[2])
    betas_g = (betas_1_g,betas_2_g)
    lambda_cyc = trial.suggest_float(name="lambda_cyc",low=LAMBDA_CYC[0],high=LAMBDA_CYC[1],step=LAMBDA_CYC[2])
    lambda_idt = trial.suggest_float(name="lambda_idt",low=LAMBDA_IDT[0],high=LAMBDA_IDT[1],step=LAMBDA_IDT[2])
    
    #fix hyperparameter
    batch_size = trial.suggest_categorical(name="batch_size",choices=BATCH_SIZE)
    epochs = trial.suggest_categorical(name="epochs",choices=EPOCHS)
    
    #cross validation
    loss_val = cross.cross_validation(trial=trial,optimizer_name=optimizer_name,lr_d=lr_d,lr_g=lr_g,betas_d=betas_d,betas_g=betas_g,batch_size=batch_size,epochs=epochs,weight_decay_d=weight_decay_d,weight_decay_g=weight_decay_g, lambda_idt=lambda_idt,lambda_cyc=lambda_cyc)
    
    return loss_val

#project 
PROJECT =  "ba-final/cyclegan-1"
API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ=="
METHOD = "GAN"

N_JOBS_LOAD_DATA = 1
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
BATCH_SIZE = [1]
EPOCHS = [100]
HIDDEN_CHANNELS_GEN = 32
HIDDEN_CHANNELS_DIS = 64
EPOCHS = [10]
n_trials = 20

fix_parameters = {"k":K,"seed":SEED,"w_size":W_SIZE,"n_trials":TOTAL_TRIALS,"batch_size":BATCH_SIZE,"epochs":EPOCHS, "hidden_channels_gen":HIDDEN_CHANNELS_GEN,"hidden_channels_dis":HIDDEN_CHANNELS_DIS}

#interval 
OPTIMIZER = ["Adam"]
LR_G = (0.00005,0.2,True)
LR_D = (0.00005,0.2,True)
WEIGHT_DECAY_G = (0.000001,0.1,True)
WEIGHT_DECAY_D = (0.000001,0.1,True)
BETAS_1_G = (0.4,0.9,0.01)
BETAS_2_G = (0.4,0.9,0.01)
BETAS_1_D = (0.4,0.999,0.01)
BETAS_2_D = (0.4,0.999,0.01)
LAMBDA_CYC = (1,20,0.1)
LAMBDA_IDT = (1,20,0.1)
fix_interval = {"optmizer":OPTIMIZER,"learning_rate_g":LR_G,"weight_decay":WEIGHT_DECAY_G,"learning_rate_d":LR_D,"weight_decay_d":WEIGHT_DECAY_D,"beta_1_d":BETAS_1_D,"beta_2_d":BETAS_2_D,"beta_1_g":BETAS_1_G,"beta_2_g":BETAS_2_G, "lambda_cyc":LAMBDA_CYC,"lambda_idt":LAMBDA_IDT}

#check device
torch.manual_seed(SEED)
np.random.seed(SEED)
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
MAE = nn.L1Loss()
MSE = nn.MSELoss()

#cross validation
cross = CrossValidation(PROJECT=PROJECT,API_TOKEN=API_TOKEN,TRANSOFRMATION=transformation,K = K,W_SIZE=W_SIZE,HIDDEN_CHANNELS_GEN=HIDDEN_CHANNELS_GEN,HIDDEN_CHANNELS_DIS=HIDDEN_CHANNELS_DIS,MAE=MAE,MSE=MSE,DEVICE=device)
dir_name = cross.dir
os.makedirs(dir_name,exist_ok=True)

#create study
storage_path = os.path.join(dir_name,"{}.db".format(METHOD))
storage_name = "sqlite:///{}".format(storage_path)

if not os.path.isfile(storage_path):
    study = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=SEED),study_name=METHOD,storage=storage_name,pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5, interval_steps=1, n_min_trials=9))
    study.optimize(objective,n_trials=n_trials)    
    
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
        