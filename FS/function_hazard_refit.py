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

def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features


# calculate cindex performance
def log_hazard_refit(x_train,df_train,x_test,df_test,x):
    
    get_target = lambda df_train: (df_train['OS'].values, df_train['OS_events'].values)
    y_train = get_target(df_train)
    
    get_target = lambda df_test: (df_test['OS'].values, df_test['OS_events'].values)
    y_test = get_target(df_test)


    
    durations_train, events_train = get_target(df_train)
    durations_test, events_test = get_target(df_test)
    
    ##------ selected features
    x_train = reduce_features(x, x_train)
    x_test = reduce_features(x, x_test)
    
    train = (x_train, y_train)  
    test = (x_test, y_test) 
    
    
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
    #verbose = False
    #callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(x_train, y_train, batch_size, epochs,verbose=False,val_data=test)
    #log = model.fit(x_train, y_train,batch_size, epochs,callbacks,verbose,val_data=valid)
    #log = model.fit(xtrain, y_train,batch_size, epochs,verbose,val_data=valid)
    #res = model.log.to_pandas()
    _ = model.compute_baseline_hazards()
    
    #--prediction-- I just included training perf too here, maybe not needed as we only care validation performance to be maximized
    surv_train = model.predict_surv_df(x_train)
    ev_train = EvalSurv(surv_train, durations_train, events_train, censor_surv='km')
    perf_train=ev_train.concordance_td('antolini')
    perf_train = round(perf_train,2)
    
    surv_test = model.predict_surv_df(x_test)
    ev_test = EvalSurv(surv_test, durations_test, events_test, censor_surv='km')
    perf_test = ev_test.concordance_td('antolini')
    perf_test = round(perf_test,2)
    
    #scoring
    risk_train = model.predict(x_train)
    risk_train = risk_train.mean(axis=1)
    
    risk_test = model.predict(x_test)
    risk_test = risk_test.mean(axis=1)
    
    return perf_train, perf_test,risk_train,risk_test
    

