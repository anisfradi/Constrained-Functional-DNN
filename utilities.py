import numpy as np
import torch as T
import random
from legendre_utils import change_domaine_legendre, eigenf_legendre
from scipy.special import eval_legendre
from scipy.interpolate import interp1d

def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    T.manual_seed(seed_value)

def take_integral(f):
    a = 1 / len(f)
    return np.trapz(f, dx=a)
    
def compute_coefs(X, j):
    L = len(X)
    x_base = np.linspace(0, 1, L)  
    y_eig = eigenf_legendre(j, x_base)
    prod = X * y_eig
    return take_integral(prod)

def interp_true_function(x, X):
    X = np.array(X)
    Scatter_true_p = X
    Scatter_number = len(Scatter_true_p)
    Scatter_X_axis = np.linspace(0, 1, num=Scatter_number, endpoint=True)
    f = interp1d(Scatter_X_axis, Scatter_true_p)
    I = take_integral(f(np.linspace(0, 1, len(X))))
    return f(x) / I

def fct_coef2function(A, PHI):
    return A @ PHI
