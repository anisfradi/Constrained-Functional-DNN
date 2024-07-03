import numpy as np
from scipy.special import legendre, eval_legendre

def eigenv_legendre(j):
    if j == 0:
        return 1
    else:
        return 1 / j / (j + 1)
        
def change_domaine_legendre(t):
    return 2*t-1
    
def eigenf_legendre(j, t):
    #change domaine, normalize, and multiply jacobi
    x = change_domaine_legendre(t)
    jacobi = np.sqrt(2)
    norm = np.sqrt(2/(2*j+1))
    return  eval_legendre(j,x)*jacobi/norm
    
def lambda_legendre_sqrt(NT):
    Lambda = np.zeros(NT)
    for j in range(0, NT):
        Lambda[j] = np.sqrt(eigenv_legendre(j))
    return np.diag(Lambda)

def Phi_Legendre(x, NT):
    #Return matrix with components [i,j] are:  eigenf_legendre(i, x_j)
    Phi = []
    for j in range(0,NT):
        Phi.append(eigenf_legendre(j , x))
    return np.asarray(Phi).T
