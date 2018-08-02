import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np

# Example of initial params to start
def get_init_params():
    params = {}
    params['objective'] = 'multiclass'
    params['num_classes'] = 9
    params['boosting_type'] = 'gbdt' # rf, dart, goss
    params['device_type'] = 'cpu'
    params['seed'] = 0

    params['num_iterations'] = 1000
    params['early_stopping_round'] = 20
    params['learning_rate'] = 0.1
    params['num_threads'] = 4

    params['num_leaves'] = 2**4
    params['min_data_in_leaf'] = 30
    params['max_depth'] = 5
    params['bagging_fraction'] = 1.0
    params['feature_fraction'] = 1.0
    params['max_bin'] = 255 # binning feature values
    params['lambda_l1'] = 0.0
    params['lambda_l2'] = 0.0
    return params

def tune_n_iterations(df, params, n_splits=5, target="target",
                      metrics=['multi_logloss'], RANDOM_STATE=42, verbose=1):
    """
    find best number of iterations by cv score
    return cv_results
    """
    skfold = StratifiedKFold(n_splits=n_splits, random_state=RANDOM_STATE)
    folds = skfold.split(X=df.drop(target,1), y=df[target].values)
    trn_ds = lgb.Dataset(df.drop(target, 1), label=df[target])
    cv_results = lgb.cv(params=params, train_set=trn_ds, folds=folds, metrics=metrics, verbose_eval=verbose)
    cv_results = pd.DataFrame(cv_results)
    return cv_results

def tune_leaf_min_data(df, params, n_iterations, params_grid, score_metric=log_loss, 
                       n_splits=10, target='target', minimize=True, RANDOM_STATE=42):
    """
    df : training dataframe
    params : parameters dict
    n_iteration : number of iterations, optimally from tune_n_iterations func
    params_grid : list of tuples having values for num_leaf and min_data
    n_splits : number of splits for stratified kfold
    target : name of target column

    returns optimal values with best score
    """
    if minimize: best_score = 1e10
    else: best_score = - 1e10
    best_params = None
    
    for num_leaf, min_data in params_grid:
        scores = []

        params['num_leaves'] = num_leaf
        params['min_data_in_leaf'] = min_data
        
        skfold = StratifiedKFold(n_splits=n_splits, random_state=RANDOM_STATE)
        folds = skfold.split(X=df.drop(target,1), y=df[target].values)
    
        for trn_idx, val_idx in folds:
            X_train, y_train = df.loc[trn_idx].drop(target, 1), df[target][trn_idx]
            X_val, y_val = df.loc[val_idx].drop(target, 1), df[target][val_idx]
            trn_ds = lgb.Dataset(X_train, y_train)
            val_ds = lgb.Dataset(X_val, y_val)
            
            model = lgb.train(params=params, train_set=trn_ds, 
                              num_boost_round=n_iterations,
                              valid_sets=[val_ds], verbose_eval=False)
            preds = model.predict(X_val)
            score = score_metric(y_val, preds)
            scores.append(score)
        
        print(f"num_leaf : {num_leaf}, min_data_in_leaf : {min_data}\t\
{score_metric.__name__}:{np.round(np.mean(scores),3), np.round(np.std(scores), 3)}")
        print()
        
        param_score = np.round(np.mean(scores),3)
        if minimize:
            if param_score < best_score:
                best_score = param_score
                best_params = [num_leaf, min_data]
        else:
            if param_score > best_score:
                best_score = param_score
                best_params = [num_leaf, min_data]
    return best_score, best_params
    

def tune_max_depth(df, params, n_iterations, params_grid, score_metric=log_loss, 
                       n_splits=10, target='target', minimize=True, RANDOM_STATE=42):
    """
    df : training dataframe
    params : parameters dict
    n_iteration : number of iterations, optimally from tune_n_iterations func
    params_grid : list of tuples having values for num_leaf and min_data
    n_splits : number of splits for stratified kfold
    target : name of target column

    returns optimal values with best score
    """
    if minimize: best_score = 1e10
    else: best_score = - 1e10
    best_params = None
    
    for max_depth in params_grid:
        scores = []

        params['max_depth'] = max_depth
        
        skfold = StratifiedKFold(n_splits=n_splits, random_state=RANDOM_STATE)
        folds = skfold.split(X=df.drop(target,1), y=df[target].values)
    
        for trn_idx, val_idx in folds:
            X_train, y_train = df.loc[trn_idx].drop(target, 1), df[target][trn_idx]
            X_val, y_val = df.loc[val_idx].drop(target, 1), df[target][val_idx]
            trn_ds = lgb.Dataset(X_train, y_train)
            val_ds = lgb.Dataset(X_val, y_val)
            
            model = lgb.train(params=params, train_set=trn_ds, 
                              num_boost_round=n_iterations,
                              valid_sets=[val_ds], verbose_eval=False)
            preds = model.predict(X_val)
            score = score_metric(y_val, preds)
            scores.append(score)
        
        print(f"max_depth : {max_depth}\t\
{score_metric.__name__}:{np.round(np.mean(scores),3), np.round(np.std(scores), 3)}")
        print()
        
        
        param_score = np.round(np.mean(scores),3)
        if minimize:
            if param_score < best_score:
                best_score = param_score
                best_params = [max_depth]
        else:
            if param_score > best_score:
                best_score = param_score
                best_params = [max_depth]
    return best_score, best_params


def tune_bag_feat_frac(df, params, n_iterations, params_grid, score_metric=log_loss, 
                       n_splits=10, target='target', minimize=True, RANDOM_STATE=42):
    """
    df : training dataframe
    params : parameters dict
    n_iteration : number of iterations, optimally from tune_n_iterations func
    params_grid : list of tuples having values for num_leaf and min_data
    n_splits : number of splits for stratified kfold
    target : name of target column

    returns optimal values with best score
    """
    if minimize: best_score = 1e10
    else: best_score = - 1e10
    best_params = None
    
    for bag_frac, feat_frac in params_grid:
        scores = []

        params['bagging_fraction'] = bag_frac
        params['feature_fraction'] = feat_frac
        
        skfold = StratifiedKFold(n_splits=n_splits, random_state=RANDOM_STATE)
        folds = skfold.split(X=df.drop(target,1), y=df[target].values)
    
        for trn_idx, val_idx in folds:
            X_train, y_train = df.loc[trn_idx].drop(target, 1), df[target][trn_idx]
            X_val, y_val = df.loc[val_idx].drop(target, 1), df[target][val_idx]
            trn_ds = lgb.Dataset(X_train, y_train)
            val_ds = lgb.Dataset(X_val, y_val)
            
            model = lgb.train(params=params, train_set=trn_ds, 
                              num_boost_round=n_iterations,
                              valid_sets=[val_ds], verbose_eval=False)
            preds = model.predict(X_val)
            score = score_metric(y_val, preds)
            scores.append(score)
        
        print(f"bagging_fraction : {bag_frac}, feature_fraction : {feat_frac}\t\
        {score_metric.__name__}:{np.round(np.mean(scores),3), np.round(np.std(scores), 3)}")
        print()
        
        
        param_score = np.round(np.mean(scores),3)
        if minimize:
            if param_score < best_score:
                best_score = param_score
                best_params = [bag_frac, feat_frac]
        else:
            if param_score > best_score:
                best_score = param_score
                best_params = [bag_frac, feat_frac]
    return best_score, best_params

def tune_l1_l2(df, params, n_iterations, params_grid, score_metric=log_loss, 
               n_splits=10, target='target', minimize=True, RANDOM_STATE=42):
    """
    df : training dataframe
    params : parameters dict
    n_iteration : number of iterations, optimally from tune_n_iterations func
    params_grid : list of tuples having values for num_leaf and min_data
    n_splits : number of splits for stratified kfold
    target : name of target column

    returns optimal values with best score
    """
    if minimize: best_score = 1e10
    else: best_score = - 1e10
    best_params = None
    
    for l1, l2 in params_grid:
        scores = []

        params['lambda_l1'] = l1
        params['lambda_l2'] = l2
        
        skfold = StratifiedKFold(n_splits=n_splits, random_state=RANDOM_STATE)
        folds = skfold.split(X=df.drop(target,1), y=df[target].values)
    
        for trn_idx, val_idx in folds:
            X_train, y_train = df.loc[trn_idx].drop(target, 1), df[target][trn_idx]
            X_val, y_val = df.loc[val_idx].drop(target, 1), df[target][val_idx]
            trn_ds = lgb.Dataset(X_train, y_train)
            val_ds = lgb.Dataset(X_val, y_val)
            
            model = lgb.train(params=params, train_set=trn_ds, 
                              num_boost_round=n_iterations,
                              valid_sets=[val_ds], verbose_eval=False)
            preds = model.predict(X_val)
            score = score_metric(y_val, preds)
            scores.append(score)
        
        print(f"L1 reg : {l1}, L2 reg : {l2}\t\
        {score_metric.__name__}:{np.round(np.mean(scores),3), np.round(np.std(scores), 3)}")
        print()
        
        param_score = np.round(np.mean(scores),3)
        if minimize:
            if param_score < best_score:
                best_score = param_score
                best_params = [l1, l2]
        else:
            if param_score > best_score:
                best_score = param_score
                best_params = [l1, l2]
    return best_score, best_params


def get_oof_preds(df, params, n_iterations, target='target', score_metric=log_loss, n_splits=10, RANDOM_STATE=42):
    """
    Get oof preds for traning data - to use in stacking
    """
    hold_out_preds = []
    hold_out_idxs = []
    scores = []    
    
    skfold = StratifiedKFold(n_splits=n_splits, random_state=RANDOM_STATE)
    folds = skfold.split(X=df.drop(target,1), y=df[target].values)
    for trn_idx, val_idx in folds:
        X_train, y_train = df.loc[trn_idx].drop(target, 1), df[target][trn_idx]
        X_val, y_val = df.loc[val_idx].drop(target, 1), df[target][val_idx]
        trn_ds = lgb.Dataset(X_train, y_train)
        val_ds = lgb.Dataset(X_val, y_val)

        model = lgb.train(params=params, train_set=trn_ds, num_boost_round=n_iterations,
                          valid_sets=[val_ds],  verbose_eval=100)
        preds = model.predict(X_val)
        
        hold_out_preds.append(preds)
        hold_out_idxs.append(val_idx)
        
        score = score_metric(y_val, preds)
        scores.append(score)
                
    print(f"{score_metric.__name__} : {np.round(np.mean(scores),3), np.round(np.std(scores), 3)}")
    hold_out_preds = np.concatenate(hold_out_preds)
    hold_out_idxs = np.concatenate(hold_out_idxs)
    train_preds_df = pd.DataFrame(hold_out_preds)
    train_preds_df.index = hold_out_idxs
    train_preds_df.sort_index(inplace=True)
    return train_preds_df