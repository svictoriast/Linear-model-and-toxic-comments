import numpy as np
from scipy import sparse
from scipy.special import expit
from time import time
from oracles import BinaryLogistic


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
            self, loss_function, step_alpha=1, step_beta=0,
            tolerance=1e-5, max_iter=1000, **kwargs
    ):
        self.history = {'time': [0.0], 'func': [], 'acc': [], 'plot_time': [0.0]}
        self.oracle = None
        self.loss_func = loss_function
        self.alpha = step_alpha
        self.beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.w = None

    def fit(self, X, y, w_0=None, trace=False, if_acc=False, X_test=None, y_test=None):
        """
        Добавлены аргументы if_acc=False, X_test=None, y_test=None
        для того, чтобы считать точность и сохранять в history
        """
        if w_0 is None:
            curr_w = np.zeros(X.shape[1])
        else:
            curr_w = w_0.copy()
        if self.loss_func == 'binary_logistic':
            self.oracle = BinaryLogistic(**self.kwargs)
        else:
            raise NotImplementedError('Only binary_logistic is implemented')
        self.w = curr_w.copy()
        if trace:
            self.history['func'].append(self.oracle.func(X, y, curr_w))
        if if_acc:
            self.history['acc'].append(np.sum(np.equal(y_test, self.predict(X_test))) / np.size(y_test))
        last_func = self.oracle.func(X, y, curr_w)
        curr_func = last_func
        k = 1
        start_time = time()
        prev_time = start_time
        while (np.abs(curr_func - last_func) >= self.tolerance and k < self.max_iter + 1) or k == 1:
            k += 1
            gw = self.oracle.grad(X, y, curr_w)
            eta = self.alpha / (k ** self.beta)
            curr_w = curr_w - eta * gw
            self.w = curr_w.copy()
            last_func = curr_func.copy()
            curr_func = self.oracle.func(X, y, curr_w)
            end_time = time()
            if trace:
                self.history['plot_time'].append(end_time - start_time)
                self.history['time'].append(end_time - prev_time)

                self.history['func'].append(curr_func)
                if if_acc:
                    self.history['acc'].append(np.sum(np.equal(y_test,
                                                               self.predict(X_test))) / np.size(y_test))

            prev_time = end_time
        self.w = curr_w.copy()
        if trace:
            return self.history

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return np.where(probs >= 0.5, 1, -1)

    def predict_proba(self, X):
        if isinstance(X, sparse.csr_matrix):
            probabil = expit(np.asarray(X @ self.w))
            return np.vstack([1 - probabil, probabil]).T
        probabil = expit(X @ self.w)
        return np.vstack([1 - probabil, probabil]).T

    def get_objective(self, X, y):
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
            self, loss_function, batch_size, step_alpha=1, step_beta=0,
            tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        self.history = {'epoch_num': [0.0], 'time': [0.0], 'func': [], 'weights_diff': [0.0], 'plot_time': [0.0],
                        'acc': []}
        self.loss_func = loss_function
        self.batch_size = batch_size
        self.alpha = step_alpha
        self.beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.kwargs = kwargs
        self.w = None
    def predict(self, X):
      return super().predict(X)
    def fit(self, X, y, w_0=None, trace=False, log_freq=1, if_acc=False, X_test=None, y_test=None):

        np.random.seed(self.random_seed)

        if w_0 is None:
            curr_w = np.zeros(X.shape[1])
        else:
            curr_w = w_0

        if self.loss_func == 'binary_logistic':
            self.oracle = BinaryLogistic(self.kwargs['l2_coef'])
        else:
            raise NotImplementedError('Only binary_logistic is implemented')
        self.w = curr_w.copy()
        if trace:
            self.history['func'].append(self.oracle.func(X, y, curr_w))
        if if_acc:
            self.history['acc'].append(np.sum(np.equal(y_test, self.predict(X_test))) / np.size(y_test))
        start_time = time()
        prev_time = start_time
        prev_loss = self.history['func'][0]
        prev_w = curr_w.copy()

        cur_loss = prev_loss + self.tolerance

        objects = 0
        flag = False

        for k in range(1, self.max_iter + 1):
            self.indexes = np.arange(X.shape[0])
            np.random.shuffle(self.indexes)
            eta = self.alpha / (k ** self.beta)
            prev_epoch = 0
            objects = 0
            for j in range(0, X.shape[0], self.batch_size):
                new_time = time()

                cur_indexes = self.indexes[j:j + self.batch_size]
                objects += len(cur_indexes)

                cur_epoch = objects / X.shape[0]
                gw = self.oracle.grad(X[cur_indexes], y[cur_indexes], curr_w)
                curr_w = curr_w - eta * gw
                self.w = curr_w.copy()

                if np.abs(cur_epoch - prev_epoch) > log_freq:

                    prev_epoch = cur_epoch
                    cur_loss = self.oracle.func(X, y, curr_w)
                    if trace:
                        self.history['time'].append(new_time - prev_time)
                        self.history['plot_time'].append(prev_time - start_time)
                        self.history['epoch_num'].append(cur_epoch)
                        self.history['weights_diff'].append((curr_w - prev_w) @ (curr_w - prev_w).T)
                        self.history['func'].append(cur_loss)
                        if if_acc:
                            self.history['acc'].append(np.sum(np.equal(y_test, self.predict(X_test))) / np.size(y_test))
                        prev_w = curr_w.copy()
                        self.w = curr_w.copy()

                    if np.abs(prev_loss - cur_loss) < self.tolerance:
                        flag = True
                        break
                prev_time = new_time
                prev_loss = cur_loss.copy()
            if flag:
                break

        self.w = curr_w.copy()
        if trace:
            return self.history
