import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.close("all")

# Make the code device-agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import E2E_DRO functions
from e2edro import e2edro as e2e
from e2edro import DataLoad as dl
from e2edro import BaseModels as bm
from e2edro import PlotFunctions as pf
from tqdm import tqdm


def Costa_training(n_z=5,n_y=40,epoches=60,data_size=[2500,500,1000],set_seed=100):

# Number of feattures and assets
    n_x=n_z
    n_y=n_y

# Number of observations per window and total number of observations
    n_obs = 100
    n_tot=data_size[0]+data_size[1]+data_size[2]
    split = [float(i)/n_tot for i in data_size ]

# Synthetic data: randomly generate data from a linear model
    X, Y = dl.synthetic_nl(n_x=n_x, n_y=n_y, n_obs=n_obs, n_tot=n_tot, split=split,set_seed=set_seed)

    perf_loss='single_period_loss'
    perf_period = 2

    # Weight assigned to MSE prediction loss function
    pred_loss_factor = 0.5

    # Risk function (default set to variance)
    prisk = 'p_var'

    # Robust decision layer to use: hellinger or tv
    dr_layer = 'hellinger'

    # Determine whether to train the prediction weights Theta
    train_pred = True

    # List of learning rates to test
    lr_list = [0.0125]

    # List of total no. of epochs to test
    epoch_list = [epoches]

    # Load saved models (default is False)
    use_cache = True

    dr_net_2layer = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, 
                    train_gamma=True, train_delta=True, pred_model='2layer',
                    set_seed=set_seed, opt_layer=dr_layer, perf_loss=perf_loss, 
                    perf_period=perf_period, pred_loss_factor=pred_loss_factor).double()
    
    dr_net_2layer.net_cv(X, Y, lr_list, epoch_list, n_val=1)

    return dr_net_2layer

def Costa_testing_score(dr_net_2layer,n_z=5,n_y=40,data_size=[2500,500,1000],set_seed=100):


    n_x=n_z
    n_y=n_y

    n_obs = 100
    n_tot=data_size[0]+data_size[1]+data_size[2]

    np.random.seed(set_seed)

    a = np.sort(np.random.rand(n_y) / 200) + 0.0005
    b = np.random.randn(n_x, n_y) / 4
    c = np.random.randn(int((n_x+1)/2), n_y)
    d = np.random.randn(n_x**2, n_y) / n_x

    # Noise std dev
    s = np.sort(np.random.rand(n_y))/20 + 0.02

    # Syntehtic features
    X = np.random.randn(n_tot, n_x) / 50
    X2 = np.random.randn(n_tot, int((n_x+1)/2)) / 50
    X_cross = 100 * (X[:,:,None] * X[:,None,:]).reshape(n_tot, n_x**2)
    X_cross = X_cross - X_cross.mean(axis=0)

    # Synthetic outputs
    Y = a + X @ b + X2 @ c + X_cross @ d + s * ( np.random.randn(n_tot, n_y) * np.abs( X ).sum(axis=1,keepdims=True))


    X=torch.from_numpy(X)
    Y=torch.from_numpy(Y)

    X_test=X[data_size[0]+data_size[1]:n_tot,:]
    Y_test=Y[data_size[0]+data_size[1]:n_tot,:]

    X_observe=X[0:n_obs]
    Y_observe=Y[0:n_obs]

    loss_rec=[]
    for i in tqdm(range(1000)):
        train_ind=torch.cat([X_observe,X_test[i,:].reshape(1,-1)],axis=0)
        z_star, y_hat=dr_net_2layer.forward(train_ind,Y_observe)
        loss_rec.append(z_star.reshape(1,40) @ Y_test[i,:].reshape(40,1))

    loss_rec_np=[]
    for i in range(1000):
        loss_rec_np.append(loss_rec[i].item())

    loss_rec_np=np.array(loss_rec_np)
    return loss_rec_np