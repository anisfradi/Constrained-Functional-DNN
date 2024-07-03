import numpy as np
import torch as T
from utilities import compute_coefs, fct_coef2function
from geometry_utils import log_map_sphere_mu

def train_set_A(NT, Data_train, nobs):
    Npts = Data_train.shape[1]
    stp = np.int8(Npts / nobs)
    idx = list(np.arange(0, Npts, stp)) + [Npts-1]
    idx = np.unique(idx)
    Train_A = np.array([compute_coefs(np.sqrt(f), i) for f in Data_train for i in range(NT)]).reshape(Data_train.shape[0], NT)
    Train_A = np.array([A / np.linalg.norm(A) for A in Train_A])
    g_obs_train = np.array([np.sqrt(f)[idx] for f in Data_train]).reshape(len(Data_train), len(idx))
    return T.tensor(g_obs_train, dtype=T.float), T.tensor(Train_A, dtype=T.float)

def test_set_A(NT, Data_test, nobs):
    Npts = Data_test.shape[1]
    stp = np.int8(Npts / nobs)
    idx = list(np.arange(0, Npts, stp)) + [Npts-1]
    idx = np.unique(idx)
    Test_A = np.array([compute_coefs(np.sqrt(f), i) for f in Data_test for i in range(NT)]).reshape(Data_test.shape[0], NT)
    Test_A = np.array([A / np.linalg.norm(A) for A in Test_A])
    g_obs_test = np.array([np.sqrt(f)[idx] for f in Data_test]).reshape(len(Data_test), len(idx))
    return T.tensor(g_obs_test, dtype=T.float), T.tensor(Test_A, dtype=T.float)
   
def train_set_B(Train_A, Pole): 
   Train_B = np.array([log_map_sphere_mu(Train_A[i], Pole) for i in range(Train_A.shape[0])])
   return T.tensor(Train_B, dtype=T.float)
