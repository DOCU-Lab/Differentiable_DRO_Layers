import torch
import numpy as np


def to_numpy(x):
    # convert torch tensor to numpy array
    return x.cpu().detach().double().numpy()


def to_torch(x, dtype, device):
    # convert numpy array to torch tensor
    return torch.from_numpy(x).type(dtype).to(device)


def synthetic_nl(n_z=5, n_y=40, data_size=[2500,500,1000], set_seed=100):
    """Generates synthetic data, adapted from "Costa, G. and Iyengar, G. N. Distributionally robust end-toend portfolio construction. Quantitative Finance, 23(10):
    1465–1482, 2023."

    Inputs
    n_z: Integer. Dimension of covariate
    n_y: Integer. Number of assets
    data_size: Number of training, validation, and test data
    set_seed: Integer.

    Outputs
    train_side,valid_side,test_side: Covariate data
    train_return,valid_return,test_return: Assets return data
    """

    training_data_size,valid_data_size,test_data_size=data_size  

    data_size=  training_data_size + valid_data_size + test_data_size
    np.random.seed(set_seed)

    a = np.sort(np.random.rand(n_y) / 200) + 0.0005
    b = np.random.randn(n_z, n_y) / 4
    c = np.random.randn(int((n_z+1)/2), n_y)
    d = np.random.randn(n_z**2, n_y) / n_z

    s = np.sort(np.random.rand(n_y))/20 + 0.02

    X = np.random.randn(data_size, n_z) / 50
    X2 = np.random.randn(data_size, int((n_z+1)/2)) / 50
    X_cross = 100 * (X[:,:,None] * X[:,None,:]).reshape(data_size, n_z**2)
    X_cross = X_cross - X_cross.mean(axis=0)

    Y = a + X @ b + X2 @ c + X_cross @ d + s * ( np.random.randn(data_size, n_y) * np.abs( X ).sum(axis=1,keepdims=True))
    X= 10 * X
    train_side=torch.from_numpy(X[0:training_data_size,:])
    valid_side=torch.from_numpy(X[training_data_size:training_data_size + valid_data_size,:])
    test_side=torch.from_numpy(X[training_data_size + valid_data_size:data_size,:])
    train_return=torch.from_numpy(Y[0:training_data_size,:])
    valid_return=torch.from_numpy(Y[training_data_size:training_data_size + valid_data_size,:])
    test_return=torch.from_numpy(Y[training_data_size + valid_data_size:data_size,:])

    return train_side,valid_side,test_side,train_return,valid_return,test_return




