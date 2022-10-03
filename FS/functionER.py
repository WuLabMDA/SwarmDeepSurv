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
from FS.function_hazard import log_hazard



# error rate
def error_rate(xdisc, ydisc, x, opts):
    ci_train,ci_valid=log_hazard(xdisc, ydisc, x, opts)
    error   = 1 - ci_valid # optimizing error on validation set to select the best mnodel
    return error,ci_train,ci_valid 
    

# Error rate & Feature size
def Fun(xdisc, ydisc,x, opts):  
    # Parameters
    alpha    = 0.99
    beta     = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
    else:
        # Get error rate
        error,ci_train,ci_valid = error_rate(xdisc, ydisc, x, opts)  #ignore ytrain
         
        # Objective function
        #cost  = alpha * error + beta * (num_feat / max_feat)
        cost  =  error * num_feat
        
    return cost,ci_train,ci_valid

