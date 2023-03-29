import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from scipy.optimize import root_scalar
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LinearRegression, LogisticRegression
from numpy.random import default_rng


def split(K, input):
    """
    #Split 1:input into K sets
    #randomly split the n samples into K folds
    #The function is use for cross-fitting
    :param K:
    :param input: length of matrix
    :return:
    """
    n = input.shape[0]
    I1 = np.zeros(n, dtype=int)
    Newids = default_rng().permutation(n, )
    m = n // K
    for i in range(K):
        I1[Newids[(i * m):(i * m + m)]] = i + 1
    return I1


def L(R, C, givenEstimator, Ctest):
    """
    Fit R ~ C by a sklearn machine learning model specified by givenEstimator
    If R is (0,1), then the output should be probability
    :param R: numpy array. R can be 0/1 variable or continuous variable
    :param C: dataframe of training data
    :param givenEstimator: any model with fit() and, depending on output type,
     predict() or predict_proba() methods
    :param Ctest: dataframe of test data
    :return: predictions on Ctest
    """
    data = C.copy()
    data['R'] = R

    # Using 5-fold to tune parameter for the machine learning model
    tt = len(pd.unique(R))

    # Fit the model. Original R code allows for hyperparameter search here, but is that fastest?
    if givenEstimator == 'logreg':
        estimator = LogisticRegression()
    elif givenEstimator == 'linreg':
        estimator = LinearRegression()
    else:
        raise ValueError
    estimator.fit(data.drop('R', axis=1), data['R'])

    if tt == 2:
        Rtest = estimator.predict_proba(Ctest)[:, 1]
    else:
        Rtest = estimator.predict(Ctest)

    return Rtest


def Lr0(Y, A, X, K, givenEstimator, Xtest):
    n = len(Y)
    I2 = split(K, X)
    Mp = np.zeros(n)
    ap = np.zeros(n)
    aBar = np.zeros(Xtest.shape[0])

    for j in range(1, K + 1):
        idNj = (I2 != j)
        idj = (I2 == j)

        # Fit hat_M^{-k,-j} = L(Y,(A,X); {-k,-j})
        df = X.copy()
        df['A'] = A
        Mp[idj] = L(Y[idNj],
                    df[idNj],
                    givenEstimator,
                    df[idj])

        # Fit hat_a^{-k,-j} = L(A,X; {-k,-j})
        atemp = L(A[idNj], X[idNj], givenEstimator, np.concatenate((X[idj], Xtest), axis=0))
        ap[idj] = atemp[0:sum(idj), ]
        aBar = aBar + atemp[sum(idj):, ]

    # Equation (3.8): aBar:= hat_a^{-k}(X^{k}) = 1/K sum_{j=1}^K hat_a^{-k,-j}(X^{k})
    aBar = aBar / K
    if np.sum(np.isinf(Mp)) > 0:
        print('Infinite in Mp')

    Mp[Mp <= 1e-8] = 1e-8
    Mp[Mp >= 1 - 1e-8] = 1 - 1e-8

    Ares2 = A - ap

    # Wi = logit(hat_M^{-k,-j}(A_i,X_i))
    Wp = logit(Mp)

    # Equation (3.7): solve beta^{-k}
    betaNk = np.sum(Ares2 * Wp) / np.sum(Ares2 ** 2)

    # t^{-k} = L(W,X;{-k}); tNk = t^{-k}(X^{k})
    tNk = L(Wp, X, givenEstimator, Xtest)

    # Defined in Equation (3.8)
    rNk = tNk - betaNk * aBar

    return rNk


def DML(Y, A, X, K, givenEstimator):
    """
    Fit the model logit(Pr(Y=1|A,X)) = beta0*A + r_0(X)
    Return a dict, with two keys: 'mXp' and 'rXp'.
    The value for 'mXp' should be the predictions on k disjoint validation sets where Y is
     predicted as a function of A and X.
     The value for 'rXp' should be the predictions on k
     disjoint validation sets where A is predicted as a function of X.

    :param Y: 1-D numpy array or pandas series for outcome
    :param A: 1-D numpy array or pandas series for treatment
    :param X: dataframe for covariates to control for
    :param K: int, for number of folds to train on
    :param givenEstimator: the machine learning to be used. Must be a model with fit() and predict() methods, a la sklearn.
    :return: dict
    """
    n = len(Y)
    mXp = np.zeros(n)
    rXp = np.zeros(n)
    I1 = split(K, Y)

    for k in range(1, K + 1):
        idNk0 = (I1 != k) & (Y == 0)
        idNk = (I1 != k)
        idk = (I1 == k)

        # Fit hat_m^{-k} = L(A,X;{-k} cap {Y==0})
        # Then obtain hat_m^{-k}(X^{-k})
        mXp[idk] = L(A[idNk0], X[idNk0], givenEstimator, X[idk])

        # Estimate hat_r^{-k} by Y^{-k}, A^{-k}, X^{-k}
        # Then obtain hat_r^{-k}(X^{k})
        rXp[idk] = Lr0(Y[idNk], A[idNk], X[idNk], K, givenEstimator, X[idk])

    return {'mXp': mXp, 'rXp': rXp}


def Estimate(Y, A, dml):
    resA = A - dml['mXp']
    C = np.sum(resA * (1 - Y) * np.exp(dml['rXp']))

    def g(beta):
        return np.sum(Y * np.exp(-beta * A) * resA) - C

    lo = 0
    up = 2
    beta0 = root_scalar(g, bracket=[lo, up]).root

    return beta0


def Bootstrap(Y, A, dml, B=1000):
    resA = A - dml['mXp']
    Betas = np.zeros(B)
    for b in range(B):
        e = np.random.normal(size=len(Y), loc=1, scale=1)
        C = np.sum(e * resA * (1 - Y) * np.exp(dml['rXp']))

        def g(beta):
            return np.sum(e * Y * np.exp(-beta * A) * resA) - C

        lo = 0
        up = 2
        try:
            beta0 = root_scalar(g, bracket=[lo, up], method='brentq').root
        except ValueError:
            beta0 = 0
        Betas[b] = beta0

    validBetas = Betas[Betas != 0]
    return np.concatenate((np.quantile(validBetas, [0.025, 0.975]),
                           [np.mean(validBetas)],
                           [np.std(validBetas)]))


def logit(x):
    x = np.log(x / (1 - x))
    if np.sum(np.isnan(x)) > 0:
        raise ValueError
    return x
