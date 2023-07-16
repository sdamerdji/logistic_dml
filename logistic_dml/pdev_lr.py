import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from pprint import pprint
from datetime import datetime
from logistic_dml import DML
from scipy import stats
from sklearn.linear_model import SGDClassifier, SGDRegressor
#from sklearn.pipeline import make_pipeline, Pipeline
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, PowerTransformer

df4 = pd.read_csv('../../dissertation/pdev/cleaned_rhna4_data.csv', low_memory=False)

train = df4.drop(['MapBlkLot_Master'], axis=1)

# Clean
train.loc[train.basement_area < 0, 'basement_area'] = np.nan
train.loc[train.n_rooms < 0, 'n_rooms'] = np.nan
train.loc[train.year_property_built > 2023, 'year_property_built'] = np.nan
train.loc[train.year_property_built < 1776, 'year_property_built'] = np.nan
train.Developed = train.Developed > 0
to_categorical = ['tax_rate_area_code', 'volume_number', 'district']
train[to_categorical] = train[to_categorical].astype(str)
train['current_sales_date_yymmdd'] = pd.to_datetime(train.current_sales_date_yymmdd, errors='coerce', format='%y/%m/%d')

# Deal with nans for log & lin reg
train = train.drop('current_sales_date_yymmdd', axis=1)
train = pd.get_dummies(train, drop_first=True, dummy_na=True)
train = train.dropna(axis=0)

Y = train.Developed
A = train.inInventory
X = train.drop(['Developed', 'inInventory'], axis=1)
X = pd.get_dummies(X, drop_first=True, dummy_na=True)

XA_rs, y_rs = RandomUnderSampler(sampling_strategy=.25,
                                 random_state=0,
                                 replacement=False).fit_resample(pd.concat((A, X), axis=1), train.Developed)
XA_rs = XA_rs.loc[:, XA_rs.nunique() > 1]
X_rs, A_rs = XA_rs.drop('inInventory', axis=1), XA_rs['inInventory']

sgc = SGDClassifier(loss='log_loss', penalty='elasticnet', max_iter=500, n_jobs=-1, learning_rate='adaptive')
sgr = SGDRegressor(penalty='elasticnet', alpha=0.01, warm_start=True, random_state=0)

lr_pipe_c = Pipeline(
    [
        ('poly', PolynomialFeatures(include_bias=False)),
        ('var', VarianceThreshold(threshold=0.01)),
        ('reduce', SelectKBest(f_regression, k=220)),
        ("scaling", QuantileTransformer(output_distribution='normal', n_quantiles=200)),
        ("logistic", sgc)
    ]
)

lr_pipe_r = Pipeline(
    [
        ('poly', PolynomialFeatures(include_bias=False)),
        ('var', VarianceThreshold(threshold=0.01)),
        ('reduce', SelectKBest(f_regression, k=220)),
        ("scaling", QuantileTransformer(output_distribution='normal', n_quantiles=200)),
        ("logistic", sgr)
    ]
)

lr_param_grid = [
    {  # Original Space
        'var__threshold': stats.loguniform(.01, 10),
        'reduce__k': stats.randint(300, 2000),
        "logistic__alpha": stats.loguniform(10**-2, 1000),
        'logistic__l1_ratio': stats.uniform(0, 1),
        'logistic__early_stopping': [True, False],
        'logistic__tol': stats.loguniform(10**-5, 10),
        'logistic__n_iter_no_change': stats.randint(2, 7),
        'logistic__eta0': stats.loguniform(.1, 10000)
    }
]

m_hat_lr_pipe = Pipeline(
    [
        ('poly', PolynomialFeatures(include_bias=False)),
        ('var', VarianceThreshold(threshold=0.265517)),
        ('reduce', SelectKBest(f_classif, k=1467)),
        ("scaling", QuantileTransformer(output_distribution='normal', n_quantiles=200)),
        ("logistic", SGDClassifier(loss='log_loss', penalty='elasticnet', max_iter=500, n_jobs=-1,
                                   learning_rate='adaptive', early_stopping=True, alpha=0.022190,
                                   eta0=10.304481, l1_ratio=0.208877, n_iter_no_change=5, tol=9.371247))
    ]
)

a_hat_lr_pipe = Pipeline(
    [
        ('poly', PolynomialFeatures(include_bias=False)),
        ('var', VarianceThreshold(threshold=4.734989)),
        ('reduce', SelectKBest(f_classif, k=899)),
        ("scaling", QuantileTransformer(output_distribution='normal', n_quantiles=200)),
        ("logistic", SGDClassifier(loss='log_loss', penalty='elasticnet', max_iter=500, n_jobs=-1,
                                   learning_rate='adaptive', early_stopping=False, alpha=5.547119,
                                   eta0=1664.672261, l1_ratio=0.857946, n_iter_no_change=3, tol=0.055129))
    ]
)

result = DML(Y=y_rs,
             A=A_rs,
             X=X_rs,
             classifier=lr_pipe_c,
             regressor=lr_pipe_r,
             k_folds=4,
             classifier_params=lr_param_grid,
             regressor_params=lr_param_grid,
             random_seed=0,
             M_hat_classifier=m_hat_lr_pipe,
             a_hat_classifier=a_hat_lr_pipe)


startTime = datetime.now()

result.train()

lb, ub, mean, sd = result.significance_testing()
print('beta hat', result.beta_hat)
print('p value', result.p_value)
print('ci', result.ci)
print('lb', lb)
print('ub', ub)
result.save('./lr_results.pkl')
print(datetime.now() - startTime)


