import numpy as np
import torch as T
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange
from metrics import GDLoss


# Define your neural network models here

class NN(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(H1, H2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(H2, H3)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(H3, D_out)

    def forward(self, x):
        y_pred = self.relu1(self.linear1(x))
        y_pred = self.relu2(self.linear2(y_pred))
        y_pred = self.relu3(self.linear3(y_pred))
        y_pred = self.linear4(y_pred) / T.norm(y_pred)  # Normalize the coefficients
        return y_pred

class NN_unnormalized(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(NN_unnormalized, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)

    def forward(self, x):
        y_pred = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(y_pred).clamp(min=0)
        y_pred = self.linear3(y_pred).clamp(min=0)
        y_pred = self.linear4(y_pred)
        return y_pred

def train_nn_model(model, train_input, train_output, criterion, optimizer, n_iterations=2500, early_stopping_delta=1e-8):
    model.train()
    training_losses = []
    for t in trange(n_iterations, desc ='Iterations', unit='iter/s'):
        y_pred = model(train_input)
        loss = criterion(y_pred, train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())
        if t > 2 and abs(training_losses[-1] - training_losses[-2]) <= early_stopping_delta:
            break
    return model

def train_nn_mse_loss(model_class, train_input, train_output, D_in, H1, H2, H3, D_out, N_train_NN=2500, delta=1e-8):
    model = model_class(D_in, H1, H2, H3, D_out)
    optimizer = Adam(model.parameters(), lr=2e-3)
    criterion = nn.MSELoss(reduction='mean')
    return train_nn_model(model, train_input, train_output, criterion, optimizer, N_train_NN, delta)
    
def train_nn_gd_loss(model_class, train_input, train_output, D_in, H1, H2, H3, D_out, lamda=0.3, N_train_NN=2500, delta=1e-8):
    model = model_class(D_in, H1, H2, H3, D_out)
    optimizer = Adam(model.parameters(), lr=2e-3)
    training_losses = []
    for t in trange(N_train_NN, desc ='Iterations', unit='iter/s'):
       y_pred = model(train_input)
       gdloss = GDLoss(y_pred, train_output)
       predicted_inner_prod = T.inner(y_pred, y_pred)
       unit_norm_penalty = T.square(T.ones_like(predicted_inner_prod) - predicted_inner_prod).trace()/y_pred.shape[0]
       loss = (1-lamda) * gdloss + lamda * unit_norm_penalty
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       training_losses.append(loss.item())
       if t>2 and np.abs(training_losses[-1]-training_losses[-2]) <= delta: 
             break
    return model

