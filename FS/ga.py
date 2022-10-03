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


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin


def roulette_wheel(prob):
    index=0  # local variable'index' refrenced b4 assigmnet error 
    num = len(prob)
    C   = np.cumsum(prob)
    P   = rand()
    for i in range(num):
        if C[i] > P:
            index = i;
            break
    
    return index


def jfs(xdisc, ydisc,run,opts):
    # Parameters
    ub       = 1
    lb       = 0
    thres    = 0.5    
    CR       = 0.8     # crossover rate
    MR       = 0.01    # mutation rate
    
    N        = opts['N']
    max_iter = opts['T']
    if 'CR' in opts:
        CR   = opts['CR'] 
    if 'MR' in opts: 
        MR   = opts['MR']  
 
     # Dimension
    dim = np.size(xdisc, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X     = init_position(lb, ub, N, dim)
    
    # Binary conversion
    X     = binary_conversion(X, thres, N, dim)
    
    #saving risks in cs file
    train_csv = pd.DataFrame()
    valid_csv = pd.DataFrame()
    #chrome_csv = pd.DataFrame()
    risk_csv = pd.DataFrame()
    result_csv= pd.DataFrame(columns=['Iter',  'Error_rate', 'Train_CI','Valid_CI', 'Subset_size'])
    

    
    # Fitness at first iteration
   
    Xgb   = np.zeros([1, dim], dtype='int')
    fitG  = float('inf')
    fit    = np.zeros([N, 3], dtype='float')
    chrome = np.zeros([1, dim], dtype='float')
    final_chr = np.zeros([max_iter, dim], dtype='float')
    Falpha = float('inf')
    Fbeta  = float('inf')
    Fdelta = float('inf')
    full_dim=xdisc.shape[1]
    
    for i in range(N):
        #fit[i,0] = Fun(xtrain, ytrain, X[i,:], opts)
        e_rate,ci_train,ci_valid= Fun(xdisc, ydisc, X[i,:], opts)
        fit[i,0]=e_rate
        fit[i,1]=ci_train
        fit[i,2]=ci_valid
        if fit[i,0] < fitG:
            Xgb[0,:] = X[i,:]
            fitG     = fit[i,0]
            Talpha      = fit[i,1]
            Valpha      = fit[i,2]
            chrome= X[i,:]

    
    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    T_curve = np.zeros([1, max_iter], dtype='float')
    V_curve = np.zeros([1, max_iter], dtype='float')
    
    t     = 0
    
    curve[0,t] = fitG.copy()
    T_curve[0,t] = Talpha.copy()
    V_curve[0,t] = Valpha.copy()
    final_chr[t,:] = chrome.copy()   # final chr
    
    
    ct=np.count_nonzero(chrome)
    result_csv = result_csv.append([pd.Series([t+1, fitG,Talpha, Valpha,ct],index = result_csv.columns[0:5])],ignore_index=True)
    print("Iteration {:2d}/{:2d} ==> error rate: {:.2f} Train CI: {:.2f} Valid CI: {:.2f} Subset Size: {:2d}/{:2d}".format( run+1,t + 1,round(curve[0,t],3), T_curve[0,t], V_curve[0,t],ct,full_dim))

    t += 1
    
    while t < max_iter:
        # Probability
        inv_fit = 1 / (1 + fit)
        prob    = inv_fit / np.sum(inv_fit) 
 
        # Number of crossovers
        Nc = 0
        for i in range(N):
            if rand() < CR:
              Nc += 1
              
        x1 = np.zeros([Nc, dim], dtype='int')
        x2 = np.zeros([Nc, dim], dtype='int')
        for i in range(Nc):
            # Parent selection
            k1      = roulette_wheel(prob)
            k2      = roulette_wheel(prob)
            P1      = X[k1,:].copy()
            P2      = X[k2,:].copy()
            # Random one dimension from 1 to dim
            index   = np.random.randint(low = 1, high = dim-1)
            # Crossover
            x1[i,:] = np.concatenate((P1[0:index] , P2[index:]))
            x2[i,:] = np.concatenate((P2[0:index] , P1[index:]))
            # Mutation
            for d in range(dim):
                if rand() < MR:
                    x1[i,d] = 1 - x1[i,d]
                    
                if rand() < MR:
                    x2[i,d] = 1 - x2[i,d]

        
        # Merge two group into one
        Xnew = np.concatenate((x1 , x2), axis=0)
        
        # Fitness
        Fnew = np.zeros([2 * Nc, 3], dtype='float')
        for i in range(2 * Nc):
            #Fnew[i,0] = Fun(xtrain, ytrain, Xnew[i,:], opts)
            e_rate,ci_train,ci_valid= Fun(xdisc, ydisc, Xnew[i,:], opts) 
            Fnew[i,0]=e_rate
            Fnew[i,1]=ci_train
            Fnew[i,2]=ci_valid
            if Fnew[i,0] < fitG:
                Xgb[0,:] = Xnew[i,:]
                fitG     = Fnew[i,0]
                Talpha      = Fnew[i,1]
                Valpha      = Fnew[i,2]
                chrome      = Xnew[i,:]
                
                
                   
        # Store result
        curve[0,t] = fitG.copy()
        T_curve[0,t] = Talpha.copy()
        V_curve[0,t] = Valpha.copy()
        final_chr[t,:] = chrome.copy()
       
        ct=np.count_nonzero(chrome)
        result_csv = result_csv.append([pd.Series([t+1, fitG,Talpha, Valpha,ct],index = result_csv.columns[0:5])],ignore_index=True)
        print("Iteration {:2d}/{:2d} ==> error rate: {:.2f} Train CI: {:.2f} Valid CI: {:.2f} Subset Size: {:2d}/{:2d}".format( run+1,t + 1,round(curve[0,t],3), T_curve[0,t], V_curve[0,t],ct,full_dim))
       
        t += 1
        
        # Elitism 
        XX  = np.concatenate((X , Xnew), axis=0)
        FF  = np.concatenate((fit , Fnew), axis=0)
        # Sort in ascending order
        ind = np.argsort(FF, axis=0)
        for i in range(N):
            X[i,:]   = XX[ind[i,0],:]
            fit[i,0] = FF[ind[i,0],0]
       
            
    # Best feature subset
    Gbin       = Xgb[0,:]
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    ga_data = {'sf': sel_index,'c': curve, 'chr':final_chr, 'nf': num_feat, 'results':result_csv}
    #ga_data = {'sf': sel_index,'c': curve, 'nf': num_feat, 'risk_train':train_csv, 'risk_valid':valid_csv,'results':result_csv,'chrome':chrome_csv}
    
    return ga_data 
            
            
                
        
        
        
    
    
    