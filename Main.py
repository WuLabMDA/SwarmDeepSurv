#================================SwarmDeepSurv================================================
#     SwarmDeepSurv: Swarm Intelligence Advances Deep Survival Network for Prognostic 
#                   Radiomics Signatures in Four Different Cancers 
#
#
# e-Mails:mbsaad@mdanderson.org, qaal@mdanderson.org, jwu11@mdanderson.org
#==================================WuLab@MDACC==============================================

import numpy as np
import pandas as pd
from FS.function_hazard_refit import log_hazard_refit
from sklearn.model_selection import train_test_split
from FS.gwo import jfs   # change this to switch algorithm 
import matplotlib.pyplot as plt
import pathlib
import os, sys
import scipy.io as sio
import warnings
warnings.simplefilter(action='ignore')



import torchtuples as tt
import torch
import torch.nn as nn

#========Splitting ========================
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=5,shuffle=False)

# -----parameter-----------
N    = 10 # number of particles/population?
T    = 30 # maximum number of iterations
opts = {'N':N, 'T':T}
chrome = pd.DataFrame()

train_csv = pd.DataFrame()
test_csv = pd.DataFrame()
final_csv= pd.DataFrame(columns=['Run','Training','Testing'])

#---load samples and survival labels-------

root_dir = pathlib.Path.cwd();
ispy_dict = np.load('my_GBM_data.npy',allow_pickle='TRUE').item()

xdisc = ispy_dict['xdisc']
xtest = ispy_dict['xtest']

ydisc = ispy_dict['ydisc']
ytest = ispy_dict['ytest']


for run in range(20):
    fmdl = jfs(xdisc, ydisc,run,opts)
    np.save('fmdl_' + str(run+1), fmdl)
    #risk_valid   = fmdl['risk_valid']
    results   = fmdl['results']
    results.to_csv('results_run' + str(run+1) +'.csv', index=False, header=True)
    chromosome=fmdl['chr']
    
    for j in range (len(chromosome)):
        chrom = chromosome[j]
        chrom = np.transpose(chrom)
        df = pd.DataFrame(chrom)
        chrome['iter_' + str(j+1)]=df
        
    chrome.to_csv('chr_run' + str(run+1) +'.csv', index=False, header=True)
    
    # testing for unseen data using the best model
    x = chrome.iloc[:,-1]
    ci_train,ci_test, risk_train, risk_test = log_hazard_refit(xdisc,ydisc,xtest,ytest,x)
    print("============================================================================ ")
    print(" Run {:2d} ==> Best model Testing CI: {:.2f} ".format(run+1,ci_test))
    print("============================================================================")
    final_csv = final_csv.append([pd.Series([run+1, ci_train,ci_test],index = final_csv.columns[0:3])],ignore_index=True)
    train_csv['Run_' + str(run+1)]=risk_train
    test_csv['Run_' + str(run+1)]=risk_test

    
train_csv.to_csv('Final_model_risk_train.csv', index=False, header=True)
test_csv.to_csv('Final_model_risk_test.csv', index=False, header=True)
final_csv.to_csv('Final_results.csv', index=False, header=True)

train_id=ydisc.index.values
train_id=pd.DataFrame(train_id)
train_id.to_csv('train_id.csv')

test_id=ytest.index.values
test_id=pd.DataFrame(test_id)
test_id.to_csv('test_id.csv')



