from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import lightgbm as lgb
import gc

def covariate_shift(df_train, df_test, feature, params):
    df_train_feature = pd.DataFrame(data={feature: df_train[feature], 'isTest': 0})
    df_test_feature = pd.DataFrame(data={feature: df_test[feature], 'isTest': 1})

    # Creating a single dataframe
    df = pd.concat([df_train_feature, df_test_feature], ignore_index=True)
    
    # Encoding if feature is categorical
    if str(df[feature].dtype) in ['object', 'category']:
        df[feature] = LabelEncoder().fit_transform(df[feature].astype(str))
    
    # Splitting it to a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(df[feature], df['isTest'], test_size=0.33, random_state=1985, stratify=df['isTest'])

    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train.values.reshape(-1, 1), y_train)
    roc_auc =  roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1])

    del df, X_train, y_train, X_test, y_test
    gc.collect();
    
    return roc_auc
    


def check_dataframe(df_train, df_test, safe_cols=[], verbose=1):
    
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    
    # check is different type from tran and test
    diff_type_cols = []
    for col in df_test.columns:
        if not df_train[col].dtype == df_test[col].dtype:
            if verbose > 0:
                print(f'train {col} is {df_train[col].dtype} and test {col} is {df_test[col].dtype}')
            diff_type_cols.append(col)
#             raise Exception(f'train {col} is {df_train[col].dtype} and test {col} is {df_test[col].dtype}')
    
    # check is number type 
    not_numbic_cols = []
    for col in df_test.columns:
        try:
#             if not np.issubdtype(df_train[col].dtype, np.number):
#             if not df_train[col].str.isnumeric():
            if not is_numeric_dtype(df_train[col]):
                if verbose > 0:
                    print(col, df_train[col].dtype)
                not_numbic_cols.append(col)
        except:
            raise Exception(f'{col}-{df_train[col].dtype}')
            
            
    # transfer object to number
    for col in not_numbic_cols:
        if verbose > 0:
            print('transfer', col)
        le = LabelEncoder()
        le.fit(df_train[col].fillna('').tolist() + df_test[col].fillna('').tolist())
        df_train[col] = le.transform(df_train[col].fillna(''))
        df_test[col] = le.transform(df_test[col].fillna(''))
    
    # check is nan
    na_existed_cols = {}
    all_na_cols = []
    for col in df_test.columns:
        train_nacount, test_nacount = df_train[col].isnull().sum(), df_test[col].isnull().sum()
        if train_nacount+test_nacount > 1:
            min_lst = []
            for df in [df_train, df_test]:
                v_ = df[col].dropna().min()
                if not np.isnan(v_):
                    min_lst.append(v_)

            max_lst = []
            for df in [df_train, df_test]:
                v_ = df[col].dropna().max()
                if not np.isnan(v_):
                    max_lst.append(v_)

            print(col, train_nacount, test_nacount, df_train[col].dtype, min_lst, max_lst)
            if len(min_lst)>1 and len(max_lst)>1:
                na_existed_cols[col] = [np.min(min_lst), np.max(max_lst)]
            else:
                all_na_cols.append(col)
    
    # check is trn tst same distribution
    trn_tst_imbalance_col = {}
    for col in df_test.columns:
        col_trn_tst_auc = covariate_shift(df_train, df_test, col, {})
        if np.abs(col_trn_tst_auc-.5) >.1:
            trn_tst_imbalance_col[col] = col_trn_tst_auc
            if verbose > 0:
                print(col, col_trn_tst_auc)
            
    # remove all nan cols
    df_train = df_train.drop(columns=all_na_cols)
    df_test = df_test.drop(columns=all_na_cols)
    
    # replace part nan value
    for col, min_max in na_existed_cols.items():
        na_replace_v = int(min_max[0]-(min_max[1]-min_max[0])/4)
        df_train[col] = df_train[col].fillna(na_replace_v)
        df_test[col] = df_test[col].fillna(na_replace_v)
    
    # remove not same distribution col
    remove_cols = [col for col in list(trn_tst_imbalance_col.keys()) if (col not in safe_cols) and (col in df_test.columns)]
    df_train = df_train.drop(columns=remove_cols)
    df_test = df_test.drop(columns=remove_cols)
            
    return df_train, df_test, diff_type_cols, not_numbic_cols, na_existed_cols, all_na_cols, trn_tst_imbalance_col