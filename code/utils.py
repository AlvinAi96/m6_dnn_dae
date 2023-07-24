"""
utils.py
@author: aihongfeng
@date: 20220309
function: utils functions.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
from scipy.stats import rankdata
from sklearn.metrics import recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



def read_dataset(root_path, train_filename, valid_filename, test_filename, pred_filename, meta_filename):
    """Load data"""
    train_data_path = root_path + train_filename
    valid_data_path = root_path + valid_filename
    test_data_path = root_path + test_filename
    pred_data_path = root_path + pred_filename
    meta_data_path = root_path + meta_filename

    train_df = pd.read_csv(train_data_path)
    valid_df = pd.read_csv(valid_data_path)
    test_df = pd.read_csv(test_data_path)
    pred_df = pd.read_csv(pred_data_path)
    meta_df = pd.read_csv(meta_data_path)
    print('train_df:{}, valid_df:{}, test_df:{}, pred_df:{}'.format(train_df.shape, valid_df.shape, test_df.shape, pred_df.shape))
    return train_df, valid_df, test_df, pred_df, meta_df



def symbol2assetID(meta_df, data_df):
    """turn asset symbol to asset ID 
    inputs：
        meta_df     (pd.DataFrame): M6_Universe.csv data
        data_df     (pd.DataFrame): original data
    returns：
        data_df     (pd.DataFrame): data with asset ID
    """
    # label encoder
    assets = meta_df['symbol'].unique() 
    asset2id_dict = {assets[i].split('.')[0]:i for i in range(len(assets))}
    data_df['asset'] = data_df['asset'].apply(lambda x:asset2id_dict[x])   
    return data_df



def add_type_id(meta_df, df):
    """add asset's type_id and subtype_id"""
    # Remove the suffix, as the suffix was removed during preprocessing
    meta_df['asset'] = meta_df['symbol'].apply(lambda x:x.split('.')[0])

    types = meta_df['GICS_sector/ETF_type'].unique() 
    type2id_dict = {types[i]:i for i in range(len(types))}
    meta_df['type_id'] = meta_df['GICS_sector/ETF_type'].apply(lambda x:type2id_dict[x])

    subtypes = meta_df['GICS_industry/ETF_subtype'].unique() 
    subtype2id_dict = {subtypes[i]:i for i in range(len(subtypes))}
    meta_df['subtype_id'] = meta_df['GICS_industry/ETF_subtype'].apply(lambda x:subtype2id_dict[x])

    new_df = pd.merge(left=df, right=meta_df[['asset', 'type_id', 'subtype_id']], how='left', left_on=['asset'], right_on=['asset'])
    return new_df



def plot_feat_importances(use_cols, model, model_method, save_path):
    """gain feature importance
    inputs：
        use_cols        (list): Features used during model training
        model           (lightgbm.basic.Booster, etc.): Trained model such as lightgbm or logistic regression
        model_method    (str): Model method, currently supports: lightgbm/logistic
        save_path       (str): Save path for feature importance plot
    returns:
        import_df       (pd.DataFrame): Feature importance
    """
    if model_method == 'lightgbm':
        import_df = pd.DataFrame()
        import_df['feature_name'] = use_cols
        import_df['importance'] = model.feature_importance(iteration=model.best_iteration)
        import_df = import_df.sort_values('importance')
        fig = plt.figure(figsize=(20,100))
        plt.barh(import_df['feature_name'], import_df['importance'])
        plt.title('LightGBM: The feature importance')
        plt.savefig(save_path + "result/lgbm_feature_importance.png") 
    elif model_method == 'logistic':
        fig = plt.figure(figsize=(150,20))
        import_dfs = []
        coefs = model.coef_
        for i in range(len(coefs)):
            import_df = pd.DataFrame()
            import_df['class'] = i+1
            import_df['feature_name'] = use_cols
            import_df['importance'] = coefs[i]
            import_df = import_df.sort_values('importance', ascending=False)
            import_dfs.append(import_df)
            plt.subplot(1,len(coefs),i+1)
            plt.barh(import_df['feature_name'], import_df['importance'])
            plt.title('Class: Rank_%s' % str(i+1))
        plt.suptitle('Logistic: The feature importance')
        plt.tight_layout()
        plt.savefig(save_path + "result/logistic_feature_importance.png")    
        import_df = pd.concat(import_dfs, axis=0)
    else:
        print("Current Model Cannot Support the Feature Importance Plotting!")    
    return import_df



def plot_rank_dist(an_asset_df, asset_id, data_type):
    """  Plot the sample distribution of the hard rank, 
    true values, and predicted values for a single asset.

    Input:
        an_asset_df       (pd.DataFrame): Dataframe with Rank1-5 and pre_Rank1-5 columns
        asset_i           (int): Asset ID
        data_type         (str): Dataset type       
    """
    RANK_DIST_COLS = ['Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5']
    PRED_RANK_DIST_COLS = ['pred_'+f for f in RANK_DIST_COLS]

    fig = plt.figure(figsize=(10,5))
    plt.subplot(3,2,1)
    plt.hist(np.argmax(an_asset_df[RANK_DIST_COLS].values, axis=1), bins=list(range(5)))
    plt.xticks(range(5), range(5))
    plt.title('%s | GT Hard Rank distribution' % data_type)
    plt.subplot(3,2,2)
    plt.hist(np.argmax(an_asset_df[PRED_RANK_DIST_COLS].values, axis=1), bins=list(range(5)))
    plt.xticks(range(5), range(5))
    plt.title('%s | Pred Hard Rank distribution' % data_type)
    plt.show()