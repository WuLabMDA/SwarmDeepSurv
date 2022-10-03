#================================SwarmDeepSurv================================================
#     SwarmDeepSurv: Swarm Intelligence Advances Deep Survival Network for Prognostic 
#                   Radiomics Signatures in Four Different Cancers 
#
#
# e-Mails:mbsaad@mdanderson.org, qaal@mdanderson.org, jwu11@mdanderson.org
#==================================WuLab@MDACC==============================================

import numpy as np
import torchtuples as tt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


from pycox.models import LogisticHazard
# from pycox.models import PMF
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH

#========Splitting ========================
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=5,shuffle=False)

#====Feature selection=====================
def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features

#==========================================
# calculate cindex performance
def log_hazard(xdisc, ydisc, x, opts):
    
    temp_ctrain=[]
    temp_cvalid=[]
    #temp_ctest=[]
    temp_array_1=[] # ID
    temp_array_2=[] # risks
    my_risk=np.empty([len(xdisc)*k_fold.n_splits,2])
    sum_risk=0
    avg_risk=[]

    
   
    for train_idx, valid_idx in k_fold.split(xdisc):
        x_train=xdisc[train_idx]
        df_train=ydisc.iloc[train_idx]
        get_target = lambda df_train: (df_train['OS'].values, df_train['OS_events'].values)
        y_train = get_target(df_train)
        
        x_valid=xdisc[valid_idx]
        df_valid=ydisc.iloc[valid_idx]
        get_target = lambda df_valid: (df_valid['OS'].values, df_valid['OS_events'].values)
        y_valid = get_target(df_valid)
        
        
        #get_target = lambda ytest: (ytest['OS'].values, ytest['OS_events'].values)
        #y_test = get_target(ytest)
        
        durations_train, events_train = get_target(df_train)
        durations_valid, events_valid = get_target(df_valid)
        #durations_test, events_test = get_target(ytest)
        
        ##------ selected features
        x_train = reduce_features(x, x_train)
        x_valid = reduce_features(x, x_valid)
        
        train = (x_train, y_train)  
        valid = (x_valid, y_valid)
        
        ##------ Training surv model------------
        in_features = x_train.shape[1] # initialize num of features
        out_features = 1  # the discrete grid number
        net = torch.nn.Sequential(
        
        torch.nn.Linear(in_features,128),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(128),
        torch.nn.Dropout(0.6),
        
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(64),
        torch.nn.Dropout(0.6),
        
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(32),
        torch.nn.Dropout(0.6),
        torch.nn.Linear(32, out_features))
        
        model = CoxPH(net, tt.optim.Adam)
        batch_size =128
        epochs = 200
        log = model.fit(x_train, y_train, batch_size, epochs,verbose=False,val_data=valid)
        _ = model.compute_baseline_hazards()
        
        #--Prediction----
        surv_train = model.predict_surv_df(x_train)
        ev_train = EvalSurv(surv_train, durations_train, events_train, censor_surv='km')
        perf_train=ev_train.concordance_td('antolini')
        perf_train = round(perf_train,2)
        temp_ctrain.append(perf_train)
        
        surv_valid = model.predict_surv_df(x_valid)
        ev_valid = EvalSurv(surv_valid, durations_valid, events_valid, censor_surv='km')
        perf_valid=ev_valid.concordance_td('antolini')
        perf_valid = round(perf_valid,2)
        temp_cvalid.append(perf_valid)
        
        #surv_test = model.predict_surv_df(xtest)
        #ev_test = EvalSurv(surv_test, durations_test, events_test, censor_surv='km')
        #perf_test=ev_test.concordance_td('antolini')
        #perf_test = round(perf_test,2)
        #temp_ctest.append(perf_test)
        

        
    ci_train=np.hstack(temp_ctrain)
    ci_valid=np.hstack(temp_cvalid)
    #ci_test=np.hstack(temp_ctest)
    
    ci_train=round(np.average(ci_train),2)
    ci_valid=round(np.average(ci_valid),2)
    #ci_test=round(np.average(ci_test),2)
    

        
    
    return ci_train, ci_valid
       
        

        
        
