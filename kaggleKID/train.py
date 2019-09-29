import pandas as pd
import numpy as np
import multiprocessing
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import gc
from time import time
import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold,TimeSeriesSplit, GroupKFold
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

import eli5
from eli5.sklearn import PermutationImportance


def lgbm_process(params, folds, df_train,df_test, columns, is_output_feature_importance=1, verbose=0):

#     aucs = list()
    his = []
    training_start_time = time()
    df_feature_importances_i_list = []
    df_valid_pred = pd.DataFrame()
    df_test_pred = pd.DataFrame()
    if type(df_test) != type(None):
        df_test_pred['TransactionID'] = df_test['TransactionID']
    
    X,y = df_train.sort_values('TransactionDT')[columns], df_train.sort_values('TransactionDT')['isFraud']
    if type(df_test) != type(None):
        X_test = df_test.sort_values('TransactionDT')[columns]
        
    for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
        start_time = time()
        if verbose > 1:
            print('Training on fold {}'.format(fold + 1))
            
        trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
        clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
        
        y_trn_pred = clf.predict(X.iloc[trn_idx].values)
        y_val_pred = clf.predict(X.iloc[test_idx].values)
        
        original_index = df_train['TransactionID'].values[test_idx]
        df_valid_pred_i = pd.DataFrame({'TransactionID': original_index, 'predict': y_val_pred, 'fold': np.zeros(y_val_pred.shape[0]) + fold})
        df_valid_pred = pd.concat([df_valid_pred, df_valid_pred_i], axis=0)
        
        y_test_pred = None
        if type(df_test)!=type(None):
            y_test_pred = clf.predict(X_test.values)
            df_test_pred_i = pd.DataFrame({fold: y_test_pred})
            df_test_pred = pd.concat([df_test_pred, df_test_pred_i], axis=1)
        
        trn_auc = roc_auc_score(y.iloc[trn_idx].values, y_trn_pred)
        val_auc = roc_auc_score(y.iloc[test_idx].values, y_val_pred)
        
        his.append({'val_auc':val_auc, 'trn_auc':trn_auc, 'y_val_pred':y_val_pred, 'y_test_pred':y_test_pred, 'test_idx':test_idx})
        
        if is_output_feature_importance:
            best_iter = clf.best_iteration
            clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)
            clf.fit(X.iloc[trn_idx].values, y.iloc[trn_idx].values)
            perm = PermutationImportance(clf, random_state=42).fit(X.iloc[test_idx].values, y.iloc[test_idx].values)
            df_feature_importances_i2 = eli5.explain_weights_dfs(perm, feature_names=columns, top=len(columns))['feature_importances']
            df_feature_importances_i2 = df_feature_importances_i2.sort_values(by=['feature'])
            df_feature_importances_i2 = df_feature_importances_i2.reset_index(drop=True)
            df_feature_importances_i_list.append(df_feature_importances_i2)
        
#         aucs.append(clf.best_score['valid_1']['auc'])
#         his.append({'val_auc':val_auc, 'trn_auc':trn_auc, 'y_val_pred':y_val_pred, 'y_test_pred':y_test_pred, 'test_idx':test_idx})
        if verbose > 0:
            print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
    
    his = pd.DataFrame(his)
    
    df_feature_importances = None
    if is_output_feature_importance:
        df_feature_importances = df_feature_importances_i_list[0]
        for idx, df_feature_importances_i in enumerate(df_feature_importances_i_list[1:]):
            df_feature_importances = pd.merge(df_feature_importances, df_feature_importances_i, on='feature', suffixes=('', idx + 1))
            
    df_valid_pred = df_valid_pred.sort_values(by=['TransactionID'])
    df_valid_pred = df_valid_pred.reset_index(drop=True)

    if type(df_test) != type(None):
        df_test_pred = df_test_pred.sort_values(by=['TransactionID'])
        df_test_pred = df_test_pred.reset_index(drop=True)
    
    if verbose > 0:
        print('-' * 30)
        print('Training has finished.')
        print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
        print('Mean AUC:', his.val_auc.mean(), his.trn_auc.mean())
        print('-' * 30)
    return his, df_feature_importances, df_valid_pred, df_test_pred, his.val_auc.mean()