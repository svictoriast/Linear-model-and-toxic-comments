import numpy as np
from scipy import sparse
from scipy.special import logsumexp, expit


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        to_exp = np.vstack((np.zeros(len(y)), -1 * X @ w * y)).T
        sum_ans = np.sum(logsumexp(to_exp, axis=1)) / len(y)
        return sum_ans + self.l2_coef * w @ w * 0.5

    def grad(self, X, y, w):
        min_val = np.finfo(np.dtype(np.float64)).min
        max_val = np.finfo(np.dtype(np.float64)).max
        if isinstance(X, sparse.csr_matrix):
            to_exp = -1 * np.asarray(X @ w) * y
            diff = np.clip(np.exp(to_exp), min_val, max_val)
            sum_answ = np.asarray(-1 * X.multiply(y[:, np.newaxis]).multiply((diff * expit(-1 * to_exp))[:, np.newaxis]).sum(axis=0))
            return np.squeeze(sum_answ, axis=0) / len(y) + self.l2_coef * w
        to_exp = -1 * X @ w * y
        diff = np.clip(np.exp(to_exp), min_val, max_val)
        sum_answ = np.sum((diff * expit(-1 * to_exp))[:, np.newaxis] * -1 * X * y[:, np.newaxis], axis=0)
        return sum_answ / len(y) + self.l2_coef * w
