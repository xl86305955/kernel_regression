from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import pairwise_kernels

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import warnings

import numpy as np
from scipy import linalg

n_samples = 3000
n_dims = 25

epsilon = 0.01

kernel = 'poly'
metric = 'linear'

def _solve_cholesky_kernel(K, y, alpha, sample_weight=None, copy=False):
    # dual_coef = inv(X X^t + alpha*Id) y
    n_samples = K.shape[0]
    #n_targets = y.shape[1]
    n_targets = 0

    if copy:
        K = K.copy()

    alpha = np.atleast_1d(alpha)
    one_alpha = (alpha == alpha[0]).all()
    has_sw = isinstance(sample_weight, np.ndarray) \
        or sample_weight not in [1.0, None]

    if has_sw:
        # Unlike other solvers, we need to support sample_weight directly
        # because K might be a pre-computed kernel.
        sw = np.sqrt(np.atleast_1d(sample_weight))
        y = y * sw[:, np.newaxis]
        K *= np.outer(sw, sw)

    if one_alpha:
        # Only one penalty, we can solve multi-target problems in one time.
        K.flat[::n_samples + 1] += alpha[0]

        try:
            # Note: we must use overwrite_a=False in order to be able to
            #       use the fall-back solution below in case a LinAlgError
            #       is raised
            dual_coef = linalg.solve(K, y, sym_pos=True,
                                     overwrite_a=False)
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in solving dual problem. Using "
                          "least-squares solution instead.")
            dual_coef = linalg.lstsq(K, y)[0]

        # K is expensive to compute and store in memory so change it back in
        # case it was user-given.
        K.flat[::n_samples + 1] -= alpha[0]

        if has_sw:
            dual_coef *= sw[:, np.newaxis]

        return dual_coef
    else:
        # One penalty per target. We need to solve each target separately.
        dual_coefs = np.empty([n_targets, n_samples], K.dtype)

        for dual_coef, target, current_alpha in zip(dual_coefs, y.T, alpha):
            K.flat[::n_samples + 1] += current_alpha

            dual_coef[:] = linalg.solve(K, target, sym_pos=True,
                                        overwrite_a=False).ravel()

            K.flat[::n_samples + 1] -= current_alpha

        if has_sw:
            dual_coefs *= sw[np.newaxis, :]

        return dual_coefs.T

def get_kernel(x, y=None):
    return pairwise_kernels(x, y, metric='linear', gamma=1)


def halfmoon(n_samples):
    """halfmoon dataset, n_samples gives the number of samples"""
    from sklearn.datasets import make_moons
    n_samples = int(n_samples)
    x, y = make_moons(n_samples=n_samples, noise=0.25,
                      random_state=0)
    trnx, tstx, trny, tsty = train_test_split(
                      x, y, test_size=1000, random_state=0)
     
    scaler = StandardScaler()
    trnx = scaler.fit_transform(trnx)
    tstx = scaler.transform(tstx)
    
    return trnx, trny, tstx, tsty

def breast_cancer():
    from sklearn.datasets import load_breast_cancer
    
    x, y = load_breast_cancer(return_X_y=True)
    trnx, tstx, trny, tsty = train_test_split(
                      x, y, test_size=200, random_state=0)
     
    scaler = StandardScaler()
    trnx = scaler.fit_transform(trnx)
    tstx = scaler.transform(tstx)
    
    return trnx, trny, tstx, tsty

def label_transform(label_arr): 
    for i in range(len(label_arr)):
        if(label_arr[i]==0):
            label_arr[i] = 1
        else:
            label_arr[i] = -1
    
    return label_arr

def acc(pred, y):
    for i in range(len(pred)):
        if(pred[i]>=0):
            pred[i] = 1
        else:
            pred[i] = -1

    miss = 0
    for i in range(len(pred)):
        if(pred[i] != y[i]):
            miss += 1

    accuracy = (len(pred) - miss) / len(pred)

    return accuracy

def adv_example(model, trnx, adv_x, adv_y, metric): 
    pred_adv = model.predict(adv_x)
    
    k_adv_x = []
    k_adv_y = []
    for i in range(len(pred_adv)):
        if(pred_adv[i] >= 0):
            if(adv_y[i] == 1):
                k_adv_x.append(adv_x[i])
                k_adv_y = np.append(k_adv_y, adv_y[i])
        else:
            if(adv_y[i] == -1):
                k_adv_x.append(adv_x[i])
                k_adv_y = np.append(k_adv_y, adv_y[i])

    k_adv_x = np.array(k_adv_x)  
    k_adv_y = np.array(k_adv_y)  
    
    idx = np.arange(len(k_adv_y))
    np.random.shuffle(idx) 
    k_adv_x = k_adv_x[idx[:len(trnx)]] 
    k_adv_y = k_adv_y[idx[:len(trnx)]] 

    n_samples = trnx.shape[0]

    k_adv_x = pairwise_kernels(k_adv_x, trnx, metric = metric)   

    g = np.zeros(shape = n_samples)
    perturb = -epsilon * (np.sign(model.dual_coef_))
    k_adv_x += perturb 

    pred_adv = _solve_cholesky_kernel(k_adv_x, k_adv_y, alpha=1)
    print('adversarial example acc:', acc(pred_adv, k_adv_y))

    return k_adv_x, k_adv_y

def train_adv_model(adv_x, model):
    adv_pred = np.dot(adv_x, model.dual_coef_)
    mean = np.mean(adv_pred)
    adv_pred -= mean
    adv_coef = _solve_cholesky_kernel(adv_x, adv_pred, alpha=1)
    
    return adv_coef

def adv_acc(adv_x, adv_coef, trnx, tstx, tsty, metric):
    k_tstx = pairwise_kernels(tstx, trnx, metric = metric)
    adv_pred = np.dot(k_tstx, adv_coef)
   
    adv_m_acc = acc(adv_pred, tsty)
    print('adversarial model acc:', adv_m_acc)


def main():
    model = KernelRidge(alpha=1, kernel=kernel)
    
#    trnx, trny, tstx, tsty = halfmoon(n_samples)
    trnx, trny, tstx, tsty = breast_cancer()

    trny = label_transform(trny)
    tsty = label_transform(tsty)

    adv_x = trnx
    adv_y = trny 

    trnx = trnx[:-200]
    trny = trny[:-200]

    model.fit(trnx, trny)

    trn_acc = acc(model.predict(trnx), trny)   
    print('training acc', trn_acc)
    tst_acc = acc(model.predict(tstx), tsty)
    print('testing acc', tst_acc)

    # Generate adversarial example
    k_adv_x, k_adv_y = adv_example(model, trnx, adv_x, adv_y, metric)

    # Build the adversarial model
    adv_coef = train_adv_model(k_adv_x, model)

    # Accuracy of the new model, training by adversarial examples
    adv_acc(k_adv_x, adv_coef, trnx, tstx, tsty, metric)


main()
