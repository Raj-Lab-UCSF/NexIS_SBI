# this file contains fns for reparameterizing SGM
import numpy as np
import torch
from scipy.stats import norm as sci_normal
from torch.distributions.normal import Normal as tor_normal

def normcdf_np(x, sd=10):
    return sci_normal(loc=0, scale=sd).cdf(x)

def normcdf_torch(x, sd=10):
    return tor_normal(loc=0, scale=sd).cdf(x)

def logistic_torch(x, k=0.10):
    """k=0.1 fits prior N(0, 100)
    """
    num = torch.exp(k*x)
    den = torch.exp(k*x) + 1
    
    # fix inf issue
    res = num/den
    res[torch.isinf(num)] = 1
    return res


def logistic_np(x, k=0.10):
    """k=0.1 fits prior N(0, 100)
    """
    num = np.exp(k*x)
    den = np.exp(k*x) + 1
    # fix inf issue
    res = num/den
    res[np.isinf(num)] = 1
    return res

def para_trans2org(paras_trans, bds, map_fn=logistic_np):
    """transform reparameterized theta to orignal theta
        args: paras_trans: an array with num_sps x num_params
              bds: an array with num_params x 2
    """
    if paras_trans.ndim == 1:
        paras_trans = paras_trans.reshape(1, -1)
    paras_org = map_fn(paras_trans)*(bds[:, 1] -  bds[:, 0]) + bds[:, 0]
    if paras_org.shape[0] == 1:
        paras_org = paras_org.reshape(-1)
    return paras_org
