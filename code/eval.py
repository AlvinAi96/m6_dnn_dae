"""
eval.py
@author: aihongfeng, liuchenning
@date: 20220419
@function: evaluation fucntion and submission checker.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')



def multi_class_eval(y_true, y_pred, data_type):
    """Print the traditional evaluation results, including overall accuracy, 
    macro/micro recall, and F1 score, for the true and predicted values of 
    the hard Rank labels of a single asset.

    Input:
        y_true      (pd.DataFrame/pd.Series/array/list): The true values of the hard Rank labels for a single asset.
        y_pred      (pd.DataFrame/pd.Series/array/list): The predicted values of the hard Rank labels for a single asset.
        data_type   (str): The type of the current dataset, e.g., train/valid/test.
    """
    acc = accuracy_score(y_true, y_pred) # Overall accuracy
    micro_recall = recall_score(y_true, y_pred, average='micro') # Micro recall
    macro_recall = recall_score(y_true, y_pred, average='macro') # Macro recall
    micro_f1 = f1_score(y_true, y_pred, average='micro') # Micro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro') # Macro F1
    print('%s |accuracy:%.3f, micro_recall:%.3f, macro_recall:%.3f, micro_f1:%.3f, macro_f1:%.3f' % (data_type, acc, micro_recall, macro_recall, micro_f1, macro_f1))



def calRPS(rank_df):
    """
    Calculate RPS_T, which represents the Rank Probability Score (RPS) for 100 assets at a single time point.

    Input:
        rank_df (pd.DataFrame): The true and predicted rank distributions for 100 assets in a single day.
    """
    rank_df = rank_df.sort_values('asset')
    pred_matrix = rank_df[['pred_Rank1', 'pred_Rank2', 'pred_Rank3', 'pred_Rank4', 'pred_Rank5']].to_numpy()
    real_matrix = rank_df[['Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5']].to_numpy() # [100, 5]
    pred_matrix = np.cumsum(pred_matrix, axis=1)
    real_matrix = np.cumsum(real_matrix, axis=1)
    # Subtract two matrices and then square the result, [100], corresponding to the RPS (Rank Probability Score) for a single asset.
    RPS_iT = np.sum((pred_matrix - real_matrix)**2, axis=1)/5   
    RPS_T = np.mean(RPS_iT) # RPS for 100 assets
    return RPS_T



def cal_overallRPS(df):
    """Calculate the overall RPS (Rank Probability Score) for 100 assets 
    across all time points, which is the average of RPS_T calculated for multiple days.

    Input:
        df (pd.DataFrame): The true and predicted rank distributions for 100 assets across multiple days.
    """
    # collect RPS_T of each day
    RPS_T_list = []
    for date in df['Date'].unique():  # forloop each day
        curr_df = df[df['Date']==date]
        curr_RPS_T = calRPS(curr_df)
        RPS_T_list.append(curr_RPS_T)

    # calc overall RPS for all days
    overallRPS = np.sum(RPS_T_list) / len(RPS_T_list)
    return overallRPS


def prepare_sub(pred_df):
    """prepare submission"""
    sub_df = pred_df[['ID','Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5']]
    sub_df['Decision'] = 0.0

    # keep 4 decimal places
    sub_df['Rank1'] = sub_df['Rank1'].apply(lambda x:round(x,4))
    sub_df['Rank2'] = sub_df['Rank2'].apply(lambda x:round(x,4))
    sub_df['Rank3'] = sub_df['Rank3'].apply(lambda x:round(x,4))
    sub_df['Rank4'] = sub_df['Rank4'].apply(lambda x:round(x,4))
    sub_df['Rank5'] = sub_df['Rank5'].apply(lambda x:round(x,4))
    # To ensure that the sum of ranks 1-5 is equal to 1, 
    # a post-processing step is performed here: unify Rank 5 as 1 minus the sum of Rank 1-4.
    sub_df['Rank5'] = 1 - sub_df[['Rank1', 'Rank2', 'Rank3', 'Rank4']].sum(axis=1)
    sub_df['Rank5'] = sub_df['Rank5'].apply(lambda x:round(x,4))
    return sub_df



def sub_checker(sub_df, meta_df):
    """
    Perform format checks on the submission results, with the following items to be checked:
    # Correctness of ID
    # Presence of 100 assets
    # Sum of rank probabilities equals 1
    # No negative values in rank
    # Sum of decisions is not greater than 1 and cannot be less than 0.25
    # Check if the predicted values in each column exceed 5 decimal places

    Input:
        root_path   (str): Root directory containing meta_fname and sub_fname.
        meta_fname  (str): Path to the provided M6_Universe.csv file.
        sub_fname   (str): Path to the submission csv file.

    Usage Example: sub_checker(root_path, meta_fname, sub_fname)
    """
    gt_asset_names = meta_df['symbol'].unique() 
    sub_asset_names = sub_df['ID'].unique()
    
    # if there are 100 assets or not 
    if len(sub_df) != 100 and len(sub_asset_names) != 100:
        print('less 100 assets!')

    # Correctness of ID and Presence of 100 assets
    miss_asset_ids = []
    for gt_asset_id in gt_asset_names:
        if gt_asset_id not in sub_asset_names:
            miss_asset_ids.append(gt_asset_id)
    if len(miss_asset_ids) != 0:
        print('cannot find IDs:', miss_asset_ids)
        print('Missing asset predictions have been automatically filled using 0.2 as the value！')
        for miss_asset_id in miss_asset_ids:
            tmp_df = pd.DataFrame({'ID':[miss_asset_id],'Rank1':[0.2], 'Rank2':[0.2], 'Rank3':[0.2], 'Rank4':[0.2], 'Rank5':[0.2], 'Decision':[0.0]})
            sub_df = pd.concat([sub_df, tmp_df], axis=0)

    # Sum of rank probabilities equals 1
    sum_prob_error_ids = []
    for sub_asset_name in sub_asset_names:
        sum_prob = np.sum(sub_df[sub_df['ID']==sub_asset_name][['Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5']].values)
        if sum_prob != 1.0:
            print(sum_prob)
            sum_prob_error_ids.append(sub_asset_name)
    if len(sum_prob_error_ids) != 0:
        print('there are IDs with their sum of rank probabilities not equals 1:', sum_prob_error_ids)
    
    # No negative values in rank
    for col in ['Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5']:
        if len(sub_df[sub_df[col]<0]) != 0:
            print('Rank has negative value!')
    
    # Sum of decisions is not greater than 1 and cannot be less than 0.25
    if sub_df['Decision'].apply(lambda x:np.abs(x)).sum() >= 0.25 and sub_df['Decision'].apply(lambda x:np.abs(x)).sum() <= 1:
        pass
    else:
        print('The sum of decisions is not within the range of [0.25, 1]!') 

    # Check if the predicted values in each column exceed 5 decimal places
    for col in ['Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5', 'Decision']:
        if sub_df[col].apply(lambda x:len(str(x).split('.')[1])).sum() > 500:
            print(col, 'exceed 5 decimal places')
    return sub_df