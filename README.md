# SwarmDeepSurv
SwarmDeepSurv: Swarm Intelligence Advances Deep Survival Network for Prognostic Radiomics Signatures in Four Solid Cancers

Submitted to Patterns - Cell Press

Swarm intelligence-based framework (termed SwarmDeepSurv) combines Swarm Intelligence (SI) with deep survival model.

SwarmDeepSurv has an advantage over traditional Cox regression and the DeepSurv because it can overcome the linearity and the hazard proportionality constrains of CoxPH and further inherits the flexible 
and robust performance from meta-heuristics SI. Particularly, two well-regarded SI algorithms namely PSO and GWO, as well as one EA algorithm named GA were integrated into DeepSurv model.
What is particularly interesting about our proposed SwarmDeepSurv is that its selected imaging features are largely different from the models with existing feature selection algorithms including the popular LASSO, strikingly with only one feature consistently selected in lung cancer. 
Thus, SwarmDeepSurv offers an alternative avenue to modeling the relationship between extracted biomarkers and clinical endpoints. 

Figure 1 illustrates overall design of the proposed SwarmDeepSurv which consists of four key phases. 
First, we started by collecting large pan-cancer imaging and demographic data from four different types of cancer, 
including lung (n=616), breast (n=186), brain (n=128), head and neck (n=128), as detailed in Supplementary Methods. 
The second phase represents the image preprocessing and radiomics feature extraction, where we extracted the standard radiomics features, 
including first order histogram features, second order textural features, and shape-based features. 
The third phase represents the model fitting through different feature selection algorithms and hyperparameter tuning, 
performed on the training cohort. Particularly, we nested the swarm intelligence with Cox proportional hazards deep neural network, i.e., DeepSurv.
Finally, the model validation stage was performed on the independent testing set for independent evaluation. 

![image](https://user-images.githubusercontent.com/94207813/193582788-58ad2f28-b3ff-4ec6-b904-9f06a717cbfa.png)


# Installation:

Download a local copy of SwarmDeepSurv and install from the directory:

git clone https://github.com/WuLabMDA/SwarmDeepSurv.git

cd SwarmDeepSurv

pip install .


# Running Experiments

You can run the main.py file

To change the swarm algorithm please go the following command: 

from FS.gwo import jfs   # change this to switch algorithm, so you can switch to PSO or GA

To change the dataset please go to the following command: 

ispy_dict = np.load('my_GBM_data.npy',allow_pickle='TRUE').item() # change to the data you want. 

To change the parameters, go the following command in th main.py file:


N    = 10 # number of particles/population?

T    = 30 # maximum number of iterations


To change the loss function, please go to the SwarmDeepSurv\FS\functionER.py


# -----sample results-----------

![image](https://user-images.githubusercontent.com/94207813/193585472-97fad366-0095-4db8-a125-9c0a9aca892d.png)


