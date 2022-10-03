#================================SwarmDeepSurv================================================
#     SwarmDeepSurv: Swarm Intelligence Advances Deep Survival Network for Prognostic 
#                   Radiomics Signatures in Four Different Cancers 
#
#
# e-Mails:mbsaad@mdanderson.org, qaal@mdanderson.org, jwu11@mdanderson.org
#==================================WuLab@MDACC==============================================

#[2014]-"Grey wolf optimizer"

import numpy as np
from numpy.random import rand
from FS.functionER import Fun
import pandas as pd


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x
    

def jfs(xdisc, ydisc,run,opts):
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5
    
    N        = opts['N']
    max_iter = opts['T']
    
    # Dimension
    dim = np.size(xdisc, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X      = init_position(lb, ub, N, dim)
    
    # Binary conversion
    Xbin   = binary_conversion(X, thres, N, dim)
    
    # Fitness at first iteration
    fit    = np.zeros([N, 3], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta  = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    chrome = np.zeros([1, dim], dtype='float')
    final_chr = np.zeros([max_iter, dim], dtype='float')
    Falpha = float('inf')
    Fbeta  = float('inf')
    Fdelta = float('inf')
    # full dimension
    full_dim=xdisc.shape[1]
    #chrome_array=[[np.empty((1,full_dim))]]
    #chrome_array=[]
    
    # saving risk in a csv file
 
    train_csv = pd.DataFrame()
    valid_csv = pd.DataFrame()
    test_csv = pd.DataFrame()
    result_csv= pd.DataFrame(columns=['Iter',  'Error_rate', 'Training','Validation', 'Subset_size'])
    

    
    for i in range(N):
        #fit[i,0] = Fun(xtrain,xvalid,Xbin[i,:], opts) ## change this!!
        e_rate,ci_train,ci_valid= Fun(xdisc, ydisc, Xbin[i,:], opts)
        fit[i,0]=e_rate
        fit[i,1]=ci_train
        fit[i,2]=ci_valid
        #fit[i,3]=ci_test
        if fit[i,0] < Falpha:
            Xalpha[0,:] = X[i,:]
            Falpha      = fit[i,0]  # error rate
            Talpha      = fit[i,1] # ci_training
            Valpha      = fit[i,2] # ci_valid
            #TTalpha      = fit[i,3] # ci_test
            chrome[0,:] = Xbin[i,:]
            
            
            
            #train_risk=risk_train
            #valid_risk=risk_valid
            #test_risk=risk_test
           
           
            
        if fit[i,0] < Fbeta and fit[i,0] > Falpha:
            Xbeta[0,:]  = X[i,:]
            Fbeta       = fit[i,0]

            
        if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
            Xdelta[0,:] = X[i,:]
            Fdelta      = fit[i,0]

   
    # Pre
    curve = np.zeros([1, max_iter], dtype='float')  # error curve
    T_curve = np.zeros([1, max_iter], dtype='float') # testing curve
    V_curve = np.zeros([1, max_iter], dtype='float')
    
    #TT_curve = np.zeros([1, max_iter], dtype='float')
    t     = 0
    
    curve[0,t] = Falpha.copy()    # error curve
    T_curve[0,t] = Talpha.copy()   # training curve
    V_curve[0,t] = Valpha.copy()   # valid curve
    final_chr[t,:] = chrome.copy()   # final chr
   # TT_curve[0,t] = TTalpha.copy()   # test curve
    
    #train_csv['iter_' + str(t+1)]=train_risk
    #valid_csv['iter_' + str(t+1)]=valid_risk
    #test_csv['iter_' + str(t+1)]=test_risk
    ct=np.count_nonzero(chrome)
   
    result_csv = result_csv.append([pd.Series([t+1, Falpha,Talpha,Valpha,ct],index = result_csv.columns[0:5])],ignore_index=True)
    
    print("Iteration {:2d}/{:2d} ==> error rate: {:.2f} Validation CI: {:.2f} Subset Size: {:2d}/{:2d}".format(run+1,t + 1,round(curve[0,t],3), V_curve[0,t],ct,full_dim))
    #chrome_array.append(temp_chr)
    
    t += 1
    
    while t < max_iter:  
      	# Coefficient decreases linearly from 2 to 0 
        a = 2 - t * (2 / max_iter) 
        
        for i in range(N):
            for d in range(dim):
                # Parameter C (3.4)
                C1     = 2 * rand()
                C2     = 2 * rand()
                C3     = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta (3.5)
                Dalpha = abs(C1 * Xalpha[0,d] - X[i,d]) 
                Dbeta  = abs(C2 * Xbeta[0,d] - X[i,d])
                Ddelta = abs(C3 * Xdelta[0,d] - X[i,d])
                # Parameter A (3.3)
                A1     = 2 * a * rand() - a
                A2     = 2 * a * rand() - a
                A3     = 2 * a * rand() - a
                # Compute X1, X2 & X3 (3.6) 
                X1     = Xalpha[0,d] - A1 * Dalpha
                X2     = Xbeta[0,d] - A2 * Dbeta
                X3     = Xdelta[0,d] - A3 * Ddelta
                # Update wolf (3.7)
                X[i,d] = (X1 + X2 + X3) / 3                
                # Boundary
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        
        # Binary conversion
        Xbin  = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            e_rate,ci_train,ci_valid= Fun(xdisc, ydisc, Xbin[i,:], opts)
            fit[i,0]=e_rate
            fit[i,1]=ci_train
            fit[i,2]=ci_valid
            #fit[i,3]=ci_test
           
            
           
            if fit[i,0] < Falpha:
                Xalpha[0,:] = X[i,:]
                Falpha      = fit[i,0]
                Talpha      =fit[i,1]  
                Valpha      =fit[i,2]
                #TTalpha      =fit[i,3]
                chrome[0,:] = Xbin[i,:]
                
               
                #train_risk=risk_train
                #valid_risk=risk_valid
                #test_risk=risk_test
                
                
            if fit[i,0] < Fbeta and fit[i,0] > Falpha:
                Xbeta[0,:]  = X[i,:]
                Fbeta       = fit[i,0]

                
            if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                Xdelta[0,:] = X[i,:]
                Fdelta      = fit[i,0]

        
        curve[0,t] = Falpha.copy()
        T_curve[0,t] = Talpha.copy()
        V_curve[0,t] = Valpha.copy()
        final_chr[t,:] = chrome.copy()
        #TT_curve[0,t] = TTalpha.copy()
        
        #train_csv['iter_' + str(t+1)]=train_risk
        #valid_csv['iter_' + str(t+1)]=valid_risk
        #test_csv['iter_' + str(t+1)]=test_risk
        ct=np.count_nonzero(chrome)
        result_csv = result_csv.append([pd.Series([t+1, Falpha,Talpha,Valpha,ct],index = result_csv.columns[0:5])],ignore_index=True)
        
        print("Iteration {:2d}/{:2d} ==> error rate: {:.2f} Validation CI: {:.2f} Subset Size: {:2d}/{:2d}".format(run+1,t + 1,round(curve[0,t],3), V_curve[0,t],ct,full_dim))
        #chrome_array.append(temp_chr)
    
        
        t += 1
        
    
                
    # Best feature subset
    Gbin       = binary_conversion(Xalpha, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    gwo_data = {'sf': sel_index,'c': curve, 'nf': num_feat,'chr':final_chr, 'results':result_csv}
    #gwo_data = {'sf': sel_index,'c': curve, 'nf': num_feat, 'risk_train':train_csv, 'risk_valid':valid_csv,'risk_test':test_csv,'results':result_csv}
    #gwo_data = {'sf': sel_index, 'chr':chrome, 'c': curve, 'nf': num_feat, 'c_train': T_curve, 'c_valid': V_curve}
    
    return gwo_data 
        
                
                
                
    
