import numpy as np
import quapy as qp
import quapy.functional as F
from scipy.special import softmax


def clr_transform(x):
    """Transformation centered log-ratio (CLR)"""
    g = np.exp(np.mean(np.log(x), axis=-1, keepdims=True))  # Media geométrica
    return np.log(x / g)


p = F.uniform_prevalence_sampling(4, 100000)

# p = p[0]
t = clr_transform(p)
mu = np.mean(t, axis=0)
cov = np.cov(t.T)
s = softmax(t, axis=-1)

print(np.all(np.isclose(p, s)))

print(mu)
print(cov)