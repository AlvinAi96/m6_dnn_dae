"""
lgb_classifier.py
@author: aihongfeng
@date: 20220419
function:
    This script trains a LGBM model based on the given training, validation, and testing sets.

    Prediction task: Multi-class classification
    Prediction target: Rank (class_num=5)
    Training target: All assets are trained together, resulting in only one model.
"""


import os
import pandas as pd
import numpy as np
import gc
import datetime
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


from utils import read_dataset, symbol2assetID, plot_feat_importances, set_seed
from feature import feature_selection
from eval import multi_class_eval, cal_overallRPS, prepare_sub, sub_checker

import lightgbm as lgb


def run(model_params,
        root_path, train_fname, valid_fname, test_fname, pred_fname, meta_fname,
        infer_flag=False,
        remove_unstable_flag=True, top_fnum=None, corr_thresh=0.10,
        save_model_flag=True, visual_import_flag=True):
    print('model_params:', model_params)

    # load data
    train_df, valid_df, test_df, pred_df, meta_df = read_dataset(root_path, train_fname, valid_fname, test_fname, pred_fname, meta_fname)
    print('Init | train_df:{}, valid_df:{}, test_df:{}, pred_df:{}'.format(train_df.shape, valid_df.shape, test_df.shape, pred_df.shape))

    # If making inference and submitting results, train using the entire dataset
    if infer_flag == True: 
        train_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)

    # define feature column and target column
    NUM_CLASS = model_params['num_class']
    TARGET_COL = 'Rank'                                                 # gt hard Rank label
    PRED_RANK_COL = 'pred_'+TARGET_COL                                  # pred hard Rank label
    RANK_DIST_COLS = [TARGET_COL+str(i) for i in range(1, NUM_CLASS+1)] # gt soft Rank label
    PRED_RANK_DIST_COLS = ['pred_'+f for f in RANK_DIST_COLS]           # pred soft Rank label
    SELF_DEFINE_USELESS_COLS = ['symbol', 'GICS_sector/ETF_type', 'type_id', 'subtype_id', 'data_type', 'return'] 

    USE_COLS = []  
    for f in list(train_df):
        if f not in ['asset', 'Date'] and \
                f != TARGET_COL and \
                f not in RANK_DIST_COLS and \
                f not in SELF_DEFINE_USELESS_COLS and \
                'ind_mean' not in f and 'ind_var' not in f:   # drop industry-agg feature
            USE_COLS.append(f)                                     

    
    # feature selection
    if remove_unstable_flag == True:
        unstable_feats = feature_selection.get_corr_unstable_feats(train_df, USE_COLS, top_fnum, corr_thresh)
        train_df.drop(unstable_feats, axis=1, inplace=True)
        valid_df.drop(unstable_feats, axis=1, inplace=True)
        test_df.drop(unstable_feats, axis=1, inplace=True)
        pred_df.drop(unstable_feats, axis=1, inplace=True)
        print('Feature Selection | train_df:{}, valid_df:{}, test_df:{}, pred_df:{}'.format(train_df.shape, valid_df.shape, test_df.shape, pred_df.shape))
        USE_COLS = [f for f in USE_COLS if f not in unstable_feats]
        
    USE_COLS.append('asset') 

    # create lgb dataset
    print('The Number of Using Features:', len(USE_COLS))
    train_matrix = lgb.Dataset(train_df[USE_COLS], label=train_df[TARGET_COL], categorical_feature=['asset'])
    valid_matrix = lgb.Dataset(valid_df[USE_COLS], label=valid_df[TARGET_COL], categorical_feature=['asset'])
    test_matrix = lgb.Dataset(test_df[USE_COLS], label=test_df[TARGET_COL], categorical_feature=['asset'])

    # train model
    if infer_flag == False: 
        lgb_model = lgb.train(model_params,
                                train_matrix,
                                num_boost_round=8,
                                valid_sets=[valid_matrix], 
                                verbose_eval=1)
    else:
        lgb_model = lgb.train(model_params,
                                train_matrix,
                                num_boost_round=8,
                                valid_sets=[test_matrix], 
                                verbose_eval=1)        

    # predict
    train_df.loc[:, PRED_RANK_DIST_COLS] = lgb_model.predict(train_df[USE_COLS], num_iteration=lgb_model.best_iteration) # gain soft-rank predictive prob distribution
    valid_df.loc[:, PRED_RANK_DIST_COLS] = lgb_model.predict(valid_df[USE_COLS], num_iteration=lgb_model.best_iteration)
    test_df.loc[:, PRED_RANK_DIST_COLS] = lgb_model.predict(test_df[USE_COLS], num_iteration=lgb_model.best_iteration)
    pred_df.loc[:, RANK_DIST_COLS] = lgb_model.predict(pred_df[USE_COLS], num_iteration=lgb_model.best_iteration)
    train_df.loc[:, PRED_RANK_COL] = np.argmax(train_df[PRED_RANK_DIST_COLS].values, axis=1) # based on distribution，gain hard-rank prediciton
    valid_df.loc[:, PRED_RANK_COL] = np.argmax(valid_df[PRED_RANK_DIST_COLS].values, axis=1)
    test_df.loc[:, PRED_RANK_COL] = np.argmax(test_df[PRED_RANK_DIST_COLS].values, axis=1) 

    # save best model
    if save_model_flag == True:
        model_path = root_path + 'model/lgb/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        lgb_model.save_model(model_path + 'lgb_classifer.txt', num_iteration=lgb_model.best_iteration)

    # plot feature importance
    if visual_import_flag == True:
        _ = plot_feat_importances(USE_COLS, lgb_model, 'lightgbm', root_path+'./')
        

    print('==========Total Result Analysis==========')
    # Calculate classification metrics for multiple-day assets.
    trn_acc, trn_micro_recall, trn_macro_recall, trn_micro_f1, trn_macro_f1 = multi_class_eval(train_df['Rank'], train_df['pred_Rank'], 'train')
    val_acc, val_micro_recall, val_macro_recall, val_micro_f1, val_macro_f1 = multi_class_eval(valid_df['Rank'], valid_df['pred_Rank'], 'valid')
    tst_acc, tst_micro_recall, tst_macro_recall, tst_micro_f1, tst_macro_f1 = multi_class_eval(test_df['Rank'], test_df['pred_Rank'], 'test')

    # Calculate rps metrics for multiple-day assets
    trn_overallRPS = cal_overallRPS(train_df)
    val_overallRPS = cal_overallRPS(valid_df)
    tst_overallRPS = cal_overallRPS(test_df)
    print('RPS | train:%.3f, valid:%.3f, test:%.3f' % (trn_overallRPS, val_overallRPS, tst_overallRPS))

    # save result
    collect_cols = ['Date', 'asset'] + [TARGET_COL] + [PRED_RANK_COL] + RANK_DIST_COLS + PRED_RANK_DIST_COLS
    result_path = root_path + 'result/lgb/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    train_df[collect_cols].to_csv(result_path + 'lgbm_pred_train_rank_df.csv', index=False)
    valid_df[collect_cols].to_csv(result_path + 'lgbm_pred_valid_rank_df.csv', index=False)
    test_df[collect_cols].to_csv(result_path + 'lgbm_pred_test_rank_df.csv', index=False)

    # prepare submission
    pred_df = pred_df.rename(columns={'symbol':'ID'})
    sub_df = prepare_sub(pred_df) 
    sub_df = sub_checker(sub_df, meta_df)  # check submission
    sub_fname = result_path + 'lgb_sub_{}_final.csv'.format(str(datetime.datetime.today()).split(' ')[0].replace('-',''))
    sub_df.to_csv(sub_fname, index=False)

if __name__=="__main__":
    set_seed(1996)
    model_params = {
    'objective':'multiclass',
    'metric':{'multiclass','multi_error'},
    'num_class':5,
    'learning_rate':0.1,
    'seed':1996,
    'boosting_type':'gbdt', # note: dart don't support early stopping
    'early_stopping_round':3,
    'colsample_bytree':0.5,
    'subsample': 0.5,
    'lambda_l1': 1.2,
    'lambda_l2': 1.2,
    'n_jobs': -1,
    'verbose':-1
    }   

    root_path = './'
    feat_type = 'BF_TZS_AMN_WSA'
    train_fname = f'pp_data2/train_rank_df_{feat_type}.csv'
    valid_fname = f'pp_data2/valid_rank_df_{feat_type}.csv'
    test_fname = f'pp_data2/test_rank_df_{feat_type}.csv'
    pred_fname = f'pp_data2/predict_rank_df_{feat_type}.csv'
    meta_fname = f'pp_data2/M6_Universe.csv'
    
    # feature selection
    top_fnum = None
    corr_thresh = 0.155

    infer_flag = False               # Is it the inference stage (i.e., training using the entire dataset)?
    remove_unstable_flag = True     # feature selection or not 
    save_model_flag = False           # save bes model ckpt or not
    visual_import_flag = True        # plot feature importance or not

    run(model_params,
        root_path, train_fname, valid_fname, test_fname, pred_fname, meta_fname,
        infer_flag,
        remove_unstable_flag,
        top_fnum,
        corr_thresh,
        save_model_flag,
        visual_import_flag)