import operator
import pandas as pd

def sort_dict(d, reverse=True):
    return sorted(d.items(), key=operator.itemgetter(1), reverse=reverse)

def col_nunique(df):
    """get number of unique elements"""
    d = {c: df[c].nunique() for c in df.columns}
    return sort_dict(d)

def col_nas(df):
    """get number of NAs"""
    d = {c :df[c].isnull().sum() for c in df.columns}
    return sort_dict(d)

def train_test_common_pairs(train, test, col1, col2):
    """
    Returns a dataframe with columns col1, col2, is_test
    and maps col1-col2 value pairs to
        - 10: pair only present in train
        - 20: pair only present in test
        - 30: pair present in both train-test
        
    e.g. col1, col2 can be item, shop pair
    """
    
    train_col1_col2 = train[[col1, col2]].drop_duplicates()
    test_col1_col2 = test[[col1, col2]].drop_duplicates()
    train_col1_col2['is_test'] = 0
    test_col1_col2['is_test'] = 1
    train_test_col1_col2 = pd.concat([train_col1_col2, test_col1_col2])
    
    train_test_col1_col2_common = train_col1_col2.merge(test_col1_col2, on=[col1, col2])[[col1, col2]]
    train_test_col1_col2_common['is_common'] = 1
    
    train_test_col1_col2_common = train_test_col1_col2.merge(train_test_col1_col2_common, on=[col1, col2], how='left')
    
    train_test_col1_col2_common.is_test = train_test_col1_col2_common.is_test.map({0:10, 1:20})
    common_idx = train_test_col1_col2_common[train_test_col1_col2_common.is_common == 1].index
    train_test_col1_col2_common.loc[common_idx, 'is_test'] = 30
    train_test_col1_col2_common = train_test_col1_col2_common.groupby([col1, col2])['is_test'].max().reset_index()
    return train_test_col1_col2_common