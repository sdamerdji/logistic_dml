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
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB

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
X_rs, A_rs = XA_rs.drop('inInventory', axis=1), XA_rs['inInventory']

"""
classification = make_pipeline(StandardScaler(),
                               SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=0.01, warm_start=True, random_state=0))
regression = make_pipeline(StandardScaler(),
                           SGDRegressor(penalty='elasticnet', alpha=0.01, warm_start=True, random_state=0))
classifier_params = {'sgdclassifier__alpha': [0.1, 0.01, 0.001],
                     'sgdclassifier__l1_ratio': [0.1, 0.5, 0.9],
                     'sgdclassifier__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                     'sgdclassifier__eta0': [0.1, 0.01],
                     'sgdclassifier__early_stopping': [True, False]}

regressor_params = {'sgdregressor__alpha': [0.1, 0.01, 0.001],
                    'sgdregressor__l1_ratio': [0.1, 0.5, 0.9],
                    'sgdregressor__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                    'sgdregressor__eta0': [0.1, 0.01],
                    'sgdregressor__early_stopping': [True, False]}
"""
classification = Pipeline([('rf', RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1))])

mhat_classifier = RandomForestClassifier(n_estimators=100,
                                         random_state=0,
                                         n_jobs=-1,
                                         min_samples_split=3,
                                         min_samples_leaf=1,
                                         max_features='log2',
                                         max_depth=13,
                                         ccp_alpha=0.9795918367346939)


regression = Pipeline([('rf', RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1))])

rf_param_grid = [
    {
        'rf__max_features': stats.uniform(.05, .25),
        'rf__min_samples_split': stats.randint(8, 20),
        'rf__min_samples_leaf': stats.randint(1, 10),
        'rf__max_depth': stats.randint(5, 10),
        'rf__ccp_alpha': stats.loguniform(1e-5, 1e-1),
        #'rf__class_weight': ['balanced', 'balanced_subsample']
    }
]

# M_Hat
M_hat_rfc_param_grid = [
    {
        'rf__max_features': stats.uniform(.3, .5),
        'rf__min_samples_split': stats.randint(8, 20),
        'rf__min_samples_leaf': stats.randint(1, 10),
        'rf__max_depth': stats.randint(5, 30),
        'rf__ccp_alpha': stats.loguniform(1e-8, 1e-2)
    }
]
"""
startTime = datetime.now()
# GaussianNB().fit(X, A)
rs = RandomizedSearchCV(classification, M_hat_rfc_param_grid, n_iter=10, random_state=0, verbose=1, cv=3)

rs.fit(X, A)
param_search_results = pd.DataFrame(rs.cv_results_['params'])
param_search_results['score'] = rs.cv_results_['mean_test_score']
param_search_results = param_search_results.sort_values('score', ascending=False)
param_search_results.to_csv('./a_hat_rf_param_search.csv')

print(param_search_results.to_string())

print(datetime.now() - startTime)

"""
result = DML(Y=y_rs,
             A=A_rs,
             X=X_rs,
             classifier=classification,
             regressor=regression,
             k_folds=5,
             classifier_params=rf_param_grid,
             regressor_params=rf_param_grid,
             random_seed=0,
             M_hat_classifier=mhat_classifier)


startTime = datetime.now()

result.train()

'''
pprint({k: np.mean(v) for k, v in result.cv_scores.items()})

scores = {k: [a for it in v for a in it['mean_test_score']] for k, v in result.cv_results.items()}
params = {k: [a for it in v for a in it['params']] for k, v in result.cv_results.items()}

for estimate in ['M_hat', 'a_hat', 't_hat']:
    print(estimate)
    print(pd.DataFrame([dict(k, **{'score': v}) for k, v in zip(params[estimate], scores[estimate])]).sort_values('score', ascending=False).to_string())
'''
lb, ub, mean, sd = result.significance_testing()
print('beta hat', result.beta_hat)
print('p value', result.p_value)
print('ci', result.ci)
print('lb', lb)
print('ub', ub)
result.save('./lr_results.pkl')
print(datetime.now() - startTime)
