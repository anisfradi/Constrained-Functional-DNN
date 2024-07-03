import numpy as np
from numpy.linalg import norm
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean

def mean_geomstat(data):
    sphere = Hypersphere(dim=data.shape[1])
    mean = FrechetMean(sphere)
    mean.fit(data)
    return mean.estimate_

def log_map_sphere_mu(A, mu):
    A = np.array(A)
    A /= norm(A)
    sphere = Hypersphere(dim=A.shape[0])
    B = sphere.metric.log(A, base_point=mu)
    return B

def exp_map_sphere_mu(B, mu):
    B = np.array(B)
    sphere = Hypersphere(B.shape[0])
    A = sphere.metric.exp(B, base_point=mu)
    return A / norm(A)
