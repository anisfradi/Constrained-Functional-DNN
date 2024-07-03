#import os  
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  
#os.environ["OMP_NUM_THREADS"] = "1"  

import numpy as np
import pandas as pd
import torch as T
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utilities import set_random_seed, take_integral, fct_coef2function
from neural_networks import train_nn_mse_loss, train_nn_gd_loss, NN, NN_unnormalized
from data_loaders import train_set_A, test_set_A, train_set_B
from legendre_utils import Phi_Legendre
from geometry_utils import mean_geomstat, exp_map_sphere_mu

set_random_seed(1)

# Load data
data = pd.read_csv('data/Beta_simulated.csv').to_numpy()

# Train-test split
train_data, test_data = train_test_split(data, test_size=0.25, random_state=0)

# Extract PDF values for training and testing
train_y = train_data[:, 2:]
test_y = test_data[:, 2:]

# Define hyperparameters
NT = 25
nobs = 30
H1, H2, H3 = 500, 1000, 200

# Preparing training and test sets
train_input, train_output = train_set_A(NT, train_y, nobs)
test_input, test_output = test_set_A(NT, test_y, nobs)

#Computing the frechet mean of A's as the pole for TMSE. 
Pole = mean_geomstat(train_output.detach().numpy()) 
Pole /= np.linalg.norm(Pole)
train_B = train_set_B(train_output.detach().numpy(), Pole)

# Training the neural network
print("Training TMSE")
model_TMSE = train_nn_mse_loss(NN_unnormalized, train_input, train_B, train_input.shape[1], H1, H2, H3, NT, N_train_NN = 500, delta = 1e-8)
print("Training PGD")
model_PGD = train_nn_gd_loss(NN_unnormalized, train_input, train_output, train_input.shape[1], H1, H2, H3, NT, lamda = 0.3, N_train_NN = 500, delta = 1e-8)

# Plot some results for visual inspection
def plot_results_PGD(model, X, y_true, n_samples=10):
    
    for i in range(n_samples):
        y_pred = model(X[i])
        y_pred = y_pred / T.norm(y_pred)
        y_pred = y_pred.detach().numpy()
        plt.figure()
        ax = np.linspace(0, 1, len(y_true[i]))
        plt.plot(ax, y_true[i], label='True')
        PHI = np.array([Phi_Legendre(i,NT) for i in ax]).T
        pred_func = fct_coef2function(y_pred, PHI)
        plt.plot(np.linspace(0, 1, len(pred_func)), pred_func**2, label='Predicted')
        plt.legend()
        plt.title("PGD-Sample {}".format(i))
        plt.show()
        
def plot_results_TMSE(model, X, y_true, n_samples=10):
    
    for i in range(n_samples):
        y_pred = model(X[i])
        y_pred = exp_map_sphere_mu(y_pred.detach().numpy(), Pole) #Exponential map to predict spherical coefficients.
        plt.figure()
        ax = np.linspace(0, 1, len(y_true[i]))
        plt.plot(ax, y_true[i], label='True')        
        PHI = np.array([Phi_Legendre(i,NT) for i in ax]).T
        pred_func = fct_coef2function(y_pred, PHI)
        plt.plot(np.linspace(0, 1, len(pred_func)), pred_func**2, label='Predicted')
        plt.legend()
        plt.title("TMSE-Sample {}".format(i))
        plt.show()


# Plot results for Beta
plot_results_PGD(model_PGD, test_input, test_y)
plot_results_TMSE(model_TMSE, test_input, test_y)


# Calculate and display some metrics
def display_metrics_PGD(model, test_input, test_output, test_y):
    MSE_metric = []
    GD_metric = []
    ISE_metric = []
    GD_f_metric = []
    PHI = np.array([Phi_Legendre(i,NT) for i in np.linspace(0,1,test_y.shape[1])]).T

    for i in range(len(test_input)):
        y_true_norm = T.norm(test_output[i])
        y_true = test_output[i] / y_true_norm if y_true_norm != 0 else test_output[i]
        y_pred = model(test_input[i])
        y_pred = y_pred / T.norm(y_pred)
        y_pred = y_pred.detach().numpy()

        measureMSE = np.linalg.norm(y_pred - y_true.numpy())
        measureGD = np.arccos(min(1, np.dot(y_true.numpy(), y_pred)))
        fpredict = fct_coef2function(y_pred, PHI)**2
        #ftrue = fct_coef2function(y_true.numpy(), lambda_legendre_sqrt(NT)) ** 2
        ftrue = test_y[i]

        inner_prod = take_integral(np.sqrt(fpredict/take_integral(fpredict)) * np.sqrt(ftrue/take_integral(ftrue)))
        GD_f = 0 if inner_prod > 1 else np.pi if inner_prod < -1 else np.arccos(inner_prod)

        GD_metric.append(measureGD)
        MSE_metric.append(measureMSE)
        ISE_metric.append(take_integral(np.square(fpredict - ftrue)))
        GD_f_metric.append(GD_f)

    print("MSE: ", np.mean(MSE_metric))
    print("GD: ", np.mean(GD_metric))
    print("ISE: ", np.mean(ISE_metric))
    print("GD_f: ", np.mean(GD_f_metric))
    
def display_metrics_TMSE(model, test_input, test_output, test_y):
    MSE_metric = []
    GD_metric = []
    ISE_metric = []
    GD_f_metric = []
    PHI = np.array([Phi_Legendre(i,NT) for i in np.linspace(0,1,test_y.shape[1])]).T

    for i in range(len(test_input)):
        y_true_norm = T.norm(test_output[i])
        y_true = test_output[i] / y_true_norm if y_true_norm != 0 else test_output[i]
        y_pred = model(test_input[i])
        y_pred = exp_map_sphere_mu(y_pred.detach().numpy(), Pole)
        measureMSE = np.linalg.norm(y_pred - y_true.numpy())
        measureGD = np.arccos(min(1, np.dot(y_true.numpy(), y_pred)))
        fpredict = fct_coef2function(y_pred, PHI)**2
        ftrue = test_y[i]

        inner_prod = take_integral(np.sqrt(fpredict/take_integral(fpredict)) * np.sqrt(ftrue/take_integral(ftrue)))
        GD_f = 0 if inner_prod > 1 else np.pi if inner_prod < -1 else np.arccos(inner_prod)

        GD_metric.append(measureGD)
        MSE_metric.append(measureMSE)
        ISE_metric.append(take_integral(np.square(fpredict - ftrue)))
        GD_f_metric.append(GD_f)

    print("MSE: ", np.mean(MSE_metric))
    print("GD: ", np.mean(GD_metric))
    print("ISE: ", np.mean(ISE_metric))
    print("GD_f: ", np.mean(GD_f_metric))

# Display metrics for Beta
print("Metrics for PGD:")
display_metrics_PGD(model_PGD, test_input, test_output, test_y)
print("Metrics for TMSE:")
display_metrics_TMSE(model_TMSE, test_input, test_output, test_y)


