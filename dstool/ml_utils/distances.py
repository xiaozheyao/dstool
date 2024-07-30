# adapted from https://github.com/awslabs/aws-cv-task2vec/blob/c5795e55ba773f9845498091a90eee2fcba5da31/task_similarity.py#L64
import numpy as np
import scipy.spatial.distance as distance

def get_variance(e, normalized=False):
    var = 1. / np.array(e.hessian)
    if normalized:
        lambda2 = 1. / np.array(e.scale)
        var = var / lambda2
    return var

def get_variances(*embeddings, normalized=False):
    return [get_variance(e, normalized=normalized) for e in embeddings]

def get_hessian(e, normalized=False):
    hess = np.array(e.hessian)
    if normalized:
        scale = np.array(e.scale)
        hess = hess / scale
    return hess

def get_hessians(*embeddings, normalized=False):
    return [get_hessian(e, normalized=normalized) for e in embeddings]


def get_scaled_hessian(e0, e1):
    h0, h1 = get_hessians(e0, e1, normalized=False)
    return h0 / (h0 + h1 + 1e-8), h1 / (h0 + h1 + 1e-8)

def kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = 0.5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = 0.5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return np.maximum(kl0, kl1).sum()

def asymmetric_kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = 0.5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = 0.5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return kl0.sum()

def jsd(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    var = 0.5 * (var0 + var1)
    kl0 = 0.5 * (var0 / var - 1 + np.log(var) - np.log(var0))
    kl1 = 0.5 * (var1 / var - 1 + np.log(var) - np.log(var1))
    return (0.5 * (kl0 + kl1)).mean()

def cosine(e0, e1):
    h1, h2 = get_scaled_hessian(e0, e1)
    return distance.cosine(h1, h2)

def normalized_cosine(e0, e1):
    h1, h2 = get_variances(e0, e1, normalized=True)
    return distance.cosine(h1, h2)

def correlation(e0, e1):
    v1, v2 = get_variances(e0, e1, normalized=False)
    return distance.correlation(v1, v2)

def binary_entropy(p):
    from scipy.special import xlogy
    return - (xlogy(p, p) + xlogy(1. - p, 1. - p))

def entropy(e0, e1):
    h1, h2 = get_scaled_hessian(e0, e1)
    return np.log(2) - binary_entropy(h1).mean()