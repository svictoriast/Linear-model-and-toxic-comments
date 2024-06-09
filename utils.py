import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    result = []
    for i in range(len(w)):
        e_i = np.zeros(len(w))
        e_i[i] = 1
        result_i = (function(w + eps * e_i) - function(w)) / eps
        result.append(result_i)
    return result