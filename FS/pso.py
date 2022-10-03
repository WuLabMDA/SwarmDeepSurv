#================================SwarmDeepSurv================================================
#     SwarmDeepSurv: Swarm Intelligence Advances Deep Survival Network for Prognostic 
#                   Radiomics Signatures in Four Different Cancers 
#
#
# e-Mails:mbsaad@mdanderson.org, qaal@mdanderson.org, jwu11@mdanderson.org
#==================================WuLab@MDACC==============================================

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


def init_velocity(lb, ub, N, dim):
    V    = np.zeros([N, dim], dtype='float')
    Vmax = np.zeros([1, dim], dtype='float')
    Vmin = np.zeros([1, dim], dtype='float')
    # Maximum & minimum velocity
    for d in range(dim):
        Vmax[0,d] = (ub[0,d] - lb[0,d]) / 2
        Vmin[0,d] = -Vmax[0,d]
        
    for i in range(N):
        for d in range(dim):
            V[i,d] = Vmin[0,d] + (Vmax[0,d] - Vmin[0,d]) * rand()
        
    return V, Vmax, Vmin


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
    w     = 0.9    # inertia weight
    c1    = 2      # acceleration factor
    c2    = 2      # acceleration factor
    
    N        = opts['N']
    max_iter = opts['T']
    if 'w' in opts:
        w    = opts['w']
    if 'c1' in opts:
        c1   = opts['c1']
    if 'c2' in opts:
        c2   = opts['c2'] 
    
    # Dimension
    dim = np.size(xdisc, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position & velocity
    X             = init_position(lb, ub, N, dim)
    V, Vmax, Vmin = init_velocity(lb, ub, N, dim) 
    
    #saving risks in cs file
    train_csv = pd.DataFrame()
    valid_csv = pd.DataFrame()
    risk_csv = pd.DataFrame()
    result_csv= pd.DataFrame(columns=['Iter',  'Error_rate', 'Train_CI','Valid_CI', 'Subset_size'])
    

    
    # Pre
    fit   = np.zeros([N, 3], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='float')
    fitG  = float('inf')
    Xpb   = np.zeros([N, dim], dtype='float')
    fitP  = float('inf') * np.ones([N, 1], dtype='float')
    curve = np.zeros([1, max_iter], dtype='float') 
    T_curve = np.zeros([1, max_iter], dtype='float')
    V_curve = np.zeros([1, max_iter], dtype='float')
    chrome = np.zeros([1, dim], dtype='float')
    final_chr = np.zeros([max_iter, dim], dtype='float')
    full_dim=xdisc.shape[1]
    
    t     = 0
    
    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            #fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
            e_rate,ci_train,ci_valid= Fun(xdisc, ydisc, Xbin[i,:], opts)
            fit[i,0]=e_rate
            fit[i,1]=ci_train
            fit[i,2]=ci_valid
            if fit[i,0] < fitP[i,0]:
                Xpb[i,:]  = X[i,:]
                fitP[i,0] = fit[i,0]

            if fitP[i,0] < fitG:
                Xgb[0,:]  = Xpb[i,:]
                fitG      = fitP[i,0]
                Talpha      = fit[i,1]
                Valpha      = fit[i,2]
                chrome[0,:] = Xbin[i,:]
                
                
                

        
        # Store result
        curve[0,t] = fitG.copy()
        T_curve[0,t] = Talpha.copy()
        V_curve[0,t] = Valpha.copy()
        final_chr[t,:] = chrome.copy()

        ct=np.count_nonzero(chrome)
        result_csv = result_csv.append([pd.Series([t+1, fitG,Talpha, Valpha,ct],index = result_csv.columns[0:5])],ignore_index=True)
        print("Iteration {:2d}/{:2d} ==> error rate: {:.2f} Train CI: {:.2f} Valid CI: {:.2f} Subset Size: {:2d}/{:2d}".format( run+1,t + 1,round(curve[0,t],3), T_curve[0,t], V_curve[0,t],ct,full_dim))
        
        t += 1
        
        for i in range(N):
            for d in range(dim):
                # Update velocity
                r1     = rand()
                r2     = rand()
                V[i,d] = w * V[i,d] + c1 * r1 * (Xpb[i,d] - X[i,d]) + c2 * r2 * (Xgb[0,d] - X[i,d]) 
                # Boundary
                V[i,d] = boundary(V[i,d], Vmin[0,d], Vmax[0,d])
                # Update position
                X[i,d] = X[i,d] + V[i,d]
                # Boundary
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
    
                
    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    pso_data = {'sf': sel_index,'c': curve, 'chr':final_chr, 'nf': num_feat,'results':result_csv}
    #pso_data = {'sf': sel_index,'c': curve, 'nf': num_feat, 'risk_train':train_csv, 'risk_valid':valid_csv,'results':result_csv}
    
    return pso_data    
    
    







