"""
feature_selection.py
@author: aihongfeng
@date: 20220419
@function: feature selection
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')



def get_corr_unstable_feats(tgt_df, use_cols, top_fnum=10, corr_thresh=0.15):
    """
    For the training set data, monthly statistics of feature correlation with Rank
    are performed. Based on the standard deviation of the correlation for each 
    feature, the top unstable features are selected for subsequent feature selection
    work.
    inputs：
        df       (pd.DataFrame): train dataset
        top_fnum          (int): top unstable feature size
        corr_thresh     (float): the std threshold for corr
        use_cols         (list): the feature list waiting for selecte
    Note: if choosing corr_thresh instead of top_fnum, you just need set top_fnum to None.
    returns：
        unstable_feats   (list): Unstable features
    """
    df = tgt_df.copy(deep=True)
    # Gain (year, month) pair 
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_year'] = df['Date'].dt.year
    df['Date_month'] = df['Date'].dt.month
    def concat_year_month(year, month):
        return (year, month)
    df['Date_ym'] = df.apply(lambda x:concat_year_month(x['Date_year'], x['Date_month']), axis=1)

    # for each month, calc feature correlation with Rank
    date_yms = df['Date_ym'].unique()
    corr_df = []
    for date_ym in date_yms:
        curr_df = df[df['Date_ym']==date_ym]
        curr_df = curr_df[[f for f in list(curr_df) if f != 'Date_ym']]
        curr_corr_df = curr_df.corr(numeric_only=True)
        curr_corr_df = curr_corr_df['Rank'].reset_index()
        curr_corr_df.rename(columns={'index':'feature', 'Rank':'corr'}, inplace=True)
        corr_df.append(curr_corr_df)
    corr_df = pd.concat(corr_df, axis=0).reset_index(drop=True)

    # select training features
    corr_df = corr_df[corr_df['feature'].isin(use_cols)]
    # print(corr_df)

    # select top unstable features based on the corr's std of each feature
    if top_fnum != None:
        top_corrStd_fnum = top_fnum 
        top_corrStd_feats = corr_df.groupby('feature').std().reset_index().sort_values('corr', ascending=False)['feature'].iloc[:top_corrStd_fnum].to_list()
    elif corr_thresh != None:
        corr_df = corr_df.groupby('feature').std().reset_index().sort_values('corr', ascending=False)
        top_corrStd_feats = corr_df[corr_df['corr'] >= corr_thresh]['feature'].to_list()
    print('Features with Unstable Correlation (size:{}):{}'.format(len(top_corrStd_feats), top_corrStd_feats))

    return top_corrStd_feats