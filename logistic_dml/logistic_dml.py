import pandas as pd
from scipy.optimize import root_scalar
from numpy.random import default_rng
import numpy as np
from scipy.special import logit
from sklearn.model_selection import RandomizedSearchCV


class DML:
    def __init__(self, Y=None, A=None, X=None, classifier=None, regressor=None, k_folds=2,
                 classifier_params=None, regressor_params=None, random_seed=None):
        self.Y = Y
        self.A = A
        self.X = X
        self.k_folds = k_folds
        self.classifier = classifier
        self.regressor = regressor
        self.dml_result = None
        self.classifier_params = classifier_params
        self.regressor_params = regressor_params

        # M hat predicts Y as function of (A, x)
        # a hat predicts A as function of X
        # t predicts logit(M_hat) as function of X
        self.cv_scores = {'M_hat': [], 'a_hat': [], 't_hat': []}
        self.cv_results = {'M_hat': [], 'a_hat': [], 't_hat': []}

        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def train(self):
        # TODO: Add some way of checking how classifiers and regressors did.
        result = self.dml(self.Y, self.A, self.X, k_folds=self.k_folds)
        self.dml_result = result

    def significance_testing(self):
        assert self.dml_result is not None
        lb, ub, mean, sd = self.bootstrap(self.Y, self.A, self.dml_result, 200)
        return lb, ub, mean, sd

    @classmethod
    def split(cls, k_folds, input, seed=None):
        """
        #Split 1:input into K sets
        #randomly split the n samples into K folds
        #The function is use for cross-fitting
        :param k_folds: int
        :param input: length of matrix
        :return:
        """
        n = input.shape[0]
        result = np.zeros(n, dtype=int)
        permuted_indices = default_rng(seed).permutation(n, )
        m = n // k_folds
        for i in range(k_folds):
            result[permuted_indices[(i * m):(i * m + m)]] = i + 1
        return result

    def ml(self, train_response, train_covariates, test_covariates, save_as=''):
        """
        Fit R ~ C by a sklearn machine learning model specified by classifier or regressors
        If R is (0,1), then the output should be probability
        :param train_response: numpy array. R can be 0/1 variable or continuous variable
        :param train_covariates: dataframe of training data
        :param test_covariates: dataframe of test data
        :param classifier: any model with fit() and predict_proba() method
        :param regressor: any model with fit() and predict() method

        :return: predictions on Ctest
        """
        assert isinstance(train_response, (np.ndarray, pd.Series))
        assert isinstance(train_covariates, pd.DataFrame)
        assert isinstance(test_covariates, pd.DataFrame)
        assert self.classifier is not None or self.regressor is not None

        if save_as:
            print('Predicting', save_as)
        data = train_covariates.copy()
        data['R'] = train_response

        # Using 5-fold to tune parameter for the machine learning model
        tt = len(pd.unique(train_response))

        unique_values = set(pd.unique(train_response))
        isClassification = unique_values == {True, False} or unique_values == {0, 1}

        # Fit the model. Original R code allows for hyperparameter search here, but is that fastest?
        if isClassification:
            if self.classifier_params:
                classifier = RandomizedSearchCV(self.classifier, self.classifier_params,
                                                n_jobs=-1, n_iter=2, random_state=self.random_seed)
            else:
                classifier = self.classifier
            classifier.fit(data.drop('R', axis=1), data['R'])
            test_predictions = classifier.predict_proba(test_covariates)[:, 1]
            if self.classifier_params:
                self.cv_scores[save_as].append(classifier.best_score_)
                self.cv_results[save_as].append(classifier.cv_results_)
        else:
            if self.classifier_params:
                regressor = RandomizedSearchCV(self.regressor, self.regressor_params,
                                               n_jobs=-1, n_iter=2, random_state=self.random_seed)
            else:
                regressor = self.regressor
            regressor.fit(data.drop('R', axis=1), data['R'])
            test_predictions = regressor.predict(test_covariates)
            if self.regressor_params:
                self.cv_scores[save_as].append(regressor.best_score_)
                self.cv_results[save_as].append(regressor.cv_results_)

        return test_predictions

    def Lr0(self, trainX, testX, trainA, trainY, k_folds=2):
        n = len(trainY)
        I2 = DML.split(k_folds, trainX)
        Mp = np.zeros(n)
        ap = np.zeros(n)
        aBar = np.zeros(testX.shape[0])

        for j in range(1, k_folds + 1):
            print('Getting predictions for subfold j =', j)
            idNj = (I2 != j)
            idj = (I2 == j)

            # Fit hat_M^{-k,-j} = L(Y,(A,X); {-k,-j})
            df = trainX.copy()
            df['A'] = trainA
            Mp[idj] = self.ml(trainY[idNj], df[idNj], df[idj], save_as='M_hat')

            # Fit hat_a^{-k,-j} = L(A,X; {-k,-j})
            atemp = self.ml(trainA[idNj], trainX[idNj], pd.concat((trainX[idj], testX)), save_as='a_hat')
            ap[idj] = atemp[0:sum(idj), ]
            aBar = aBar + atemp[sum(idj):, ]

        # Equation (3.8): aBar:= hat_a^{-k}(X^{k}) = 1/K sum_{j=1}^K hat_a^{-k,-j}(X^{k})
        aBar = aBar / k_folds
        if np.sum(np.isinf(Mp)) > 0:
            print('Infinite in Mp')

        Mp[Mp <= 1e-8] = 1e-8
        Mp[Mp >= 1 - 1e-8] = 1 - 1e-8

        Ares2 = trainA - ap

        # Wi = logit(hat_M^{-k,-j}(A_i,X_i))
        Wp = logit(Mp)

        # Equation (3.7): solve beta^{-k}
        betaNk = np.sum(Ares2 * Wp) / np.sum(Ares2 ** 2)

        # t^{-k} = L(W,X;{-k}); tNk = t^{-k}(X^{k})
        tNk = self.ml(Wp, trainX, testX, save_as='t_hat')

        # Defined in Equation (3.8)
        rNk = tNk - betaNk * aBar

        return rNk

    def dml(self, Y, A, X, k_folds=5):
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
        :param k_folds: int, for number of folds to train on
        :return: dict
        """
        n = len(Y)
        mXp = np.zeros(n)
        rXp = np.zeros(n)
        I1 = DML.split(k_folds, Y)

        for k in range(1, k_folds + 1):
            print('Getting predictions for fold k =', k)
            idNk0 = (I1 != k) & (Y == 0)
            idNk = (I1 != k)
            idk = (I1 == k)

            # Fit hat_m^{-k} = L(A,X;{-k} cap {Y==0})
            # Then obtain hat_m^{-k}(X^{-k})
            mXp[idk] = self.ml(A[idNk0], X[idNk0], X[idk], save_as='M_hat')

            # Estimate hat_r^{-k} by Y^{-k}, A^{-k}, X^{-k}
            # Then obtain hat_r^{-k}(X^{k})
            rXp[idk] = self.Lr0(X[idNk], X[idk], A[idNk], Y[idNk], k_folds)

        return {'mXp': mXp, 'rXp': rXp}

    def estimate_beta(self, Y, A, dml):
        resA = A - dml['mXp']
        clipped = np.clip(dml['rXp'], -np.inf, 100)
        C = np.sum(resA * (1 - Y) * np.exp(clipped))

        def g(beta):
            return np.sum(Y * np.exp(-beta * A) * resA) - C

        lo = -5
        up = 5
        beta0 = root_scalar(g, bracket=[lo, up]).root

        return beta0

    def bootstrap(self, Y, A, dml, B=1000):
        resA = A - dml['mXp']
        clipped = np.clip(dml['rXp'], -np.inf, 50)
        betas = []
        cant_solve = 0
        for b in range(B):
            e = self.rng.normal(size=len(Y), loc=1, scale=1)
            C = np.sum(e * resA * (1 - Y) * np.exp(clipped))

            def g(beta):
                return np.sum(e * Y * np.exp(-beta * A) * resA) - C

            # Limits of [-5, 5] allow Beta to range from multiplier of >100x to <0.01x on odds ratio
            # I adjust to allow for larger range to allow for shitty modelling
            lo = -100
            up = 100
            try:
                beta0 = root_scalar(g, bracket=[lo, up], method='brentq').root
                betas.append(beta0)
            except ValueError:
                cant_solve += 1

        #if cantSolve > (0.01 * B):
        #    print(f'Solver failed to find a solution for {100*cantSolve/B}% bootstraps.'
        #          f' Consider changing numerical optimization.')
        assert betas, "All optimizations failed"
        betas = np.array(betas)
        return np.concatenate((np.quantile(betas, [0.025, 0.975]),
                               [np.mean(betas)],
                               [np.std(betas)]))

