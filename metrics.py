import numpy as np
import torch as T

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().item()

def gmm_mse_loss(x_compute, alpha, mu, sigma, y):
    ONEOVERSQRT2PI = 1.0 / T.sqrt(T.tensor(2 * np.pi))
    n_samples, n_gaussians, n_compute = mu.shape
    x = x_compute.expand(n_samples, n_gaussians, -1)
    mu = mu.expand(n_compute, -1, -1).permute(1, 2, 0)
    sigma = sigma.expand(n_compute, -1, -1).permute(1, 2, 0)
    alpha = alpha.expand(n_compute, -1, -1).permute(1, 2, 0)
    X = -(x - mu) ** 2 / (2 * sigma)
    X = (X * T.reciprocal(T.sqrt(sigma))) * ONEOVERSQRT2PI
    y_pred = T.sum(alpha * T.exp(X), dim=1)
    criterion = nn.MSELoss(reduction='mean')
    return criterion(y_pred, y)

def GDLoss(output, target):
    Output = (T.reciprocal(T.norm(output, dim=1)) * output.T).T  # Normalize
    Target = (T.reciprocal(T.norm(target, dim=1)) * target.T).T  # Normalize
    loss = (T.arccos(T.inner(Output, Target)) ** 2).trace()
    return loss / (len(output) + 1)
