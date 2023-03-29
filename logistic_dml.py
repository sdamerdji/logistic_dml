import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np


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
    data = pd.concat([pd.Series(R), C], axis=1)
    data.columns = ['R'] + list(C.columns)

    # Using 5-fold to tune parameter for the machine learning model
    tt = len(pd.unique(R))

    # Fit the model. Original R code allows for hyperparameter search here, but is that fastest?
    givenEstimator.fit(data.iloc[:, 1:], data['R'])

    if tt == 2:
        Rtest = givenEstimator.predict_proba(Ctest)[:, 1]
    else:
        Rtest = givenEstimator.predict(Ctest)

    return Rtest


def Lr0(Y, A, X, K, givenEstimator, Xtest):
    n = len(Y)
    I2 = Split(K, n)
    Mp = np.zeros(n)
    ap = np.zeros(n)
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

    return (rNk)


def DML(Y, A, X, K, givenEstimator):
    n = len(Y)
    mXp = np.zeros(n)
    rXp = np.zeros(n)
    I1 = Split(K, n)

    for k in range(1, K + 1):
        idNk0 = (I1 != k) & (Y == 0)
        idNk = (I1 != k)
        idk = (I1 == k)

        # Fit hat_m^{-k} = L(A,X;{-k} cap {Y==0})
        # Then obtain hat_m^{-k}(X^{-k})
        mXp[idk] = L(A[idNk0], X[idNk0, :], givenEstimator, X[idk, :])

        # Estimate hat_r^{-k} by Y^{-k}, A^{-k}, X^{-k}
        # Then obtain hat_r^{-k}(X^{k})
        rXp[idk] = Lr0(Y[idNk], A[idNk], X[idNk, :], K, givenEstimator, X[idk, :])

    return {'mXp': mXp, 'rXp': rXp}


def Estimate(Y, A, dml):
    resA = A - dml['mXp']
    C = np.sum(resA * (1 - Y) * np.exp(dml['rXp']))

    def g(beta):
        return np.sum(Y * np.exp(-beta * A) * resA) - C

    lo = 0
    up = 2
    beta0 = optimize.root_scalar(g, bracket=[lo, up]).root

    return beta0


def Bootstrap(Y, A, dml, B=1000):
    resA = A - dml['mXp']
    Betas = np.zeros(B)
    for b in range(B):
        e = np.random.normal(size=len(Y))
        C = np.sum(e * resA * (1 - Y) * np.exp(dml['rXp']))

        def g(beta):
            return np.sum(e * Y * np.exp(-beta * A) * resA) - C

        lo = 0
        up = 2
        beta0 = optimize.root_scalar(g, bracket=[lo, up]).root
        Betas[b] = beta0

    return np.concatenate(
        (np.quantile(Betas[Betas != 0], [0.025, 0.975]), np.mean(Betas[Betas != 0]), np.std(Betas[Betas != 0])))


def logit(x):
    x = np.log(x / (1 - x))
    if np.sum(np.isnan(x)) > 0:
        raise ValueError
    return x
