#coding=utf-8
"""
feature_enginner.py
@author: aihongfeng
@date: 20220418
@function: feature engineering.
"""
import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')



def load_all_files(data_dir):
    """Load all asset dataframes"""
    df_list = [] 
    for i, _, k in os.walk(data_dir):
        if len(k) !=0:  # file list is not empty
            if i[-1] != '/':
                i = i + '/'    
            for a_file in k:                       
                a_full_path = i + a_file
                a_df = pd.read_csv(a_full_path)
                a_df['asset'] = a_file.split('.csv')[0] # will be label encode
                a_df['symbol'] = a_file.split('.csv')[0]
                a_df['Date'] = pd.to_datetime(a_df['Date'])
                df_list.append(a_df)
    all_df = pd.concat(df_list, axis=0)
    return all_df


def get_basic_feats(df):
    """Expand basic features"""
    df['cash_flow'] = df['Adjusted_close'] * df['Volume']
    df['open_high_ratio'] = (df['Open'] - df['High'])/df['Open']
    df['open_low_ratio'] = (df['Open'] - df['Low'])/df['Open']
    df['close_high_ratio'] = (df['Close'] - df['High'])/df['Close']
    df['close_low_ratio'] = (df['Close'] - df['Low'])/df['Close']
    df['high_low_ratio'] = (df['High'] - df['Low'])/df['Low']
    return df


def drop_date_with_less_asset(df, asset_num=80):
    """remove the day with less asset count, otherwise ranking will be affected"""
    date2asset_df = df[['Date','asset']].drop_duplicates()
    date2asset_df = date2asset_df.groupby('Date')['asset'].count().reset_index().sort_values(by='asset')
    drop_dates = date2asset_df[date2asset_df['asset']<=asset_num]['Date']
    df = df[~df['Date'].isin(drop_dates)] 
    return df


def add_data_type(all_df, train_ratio=0.7, test_cnt=160):
    """Splite dataset based on date
    inputs:
        all_df          (pd.DataFrame)：input dataframe
        train_raio      (float): Excluding the test set, the ratio of the training set.
        test_cnt        (int): The length of the test set in terms of dates"""
    # Calculate the length of each splited dataset in terms of dates.
    all_dates = all_df['Date'].unique()
    date_len = len(all_dates)                            # total date length
    train_cnt = int((date_len - test_cnt) * train_ratio) # train date length
    valid_cnt = date_len - train_cnt - test_cnt          # valid date length

    # time series split dataset
    train_df = all_df[all_df['Date']<=all_dates[train_cnt]]
    valid_df = all_df[(all_df['Date']>all_dates[train_cnt]) & (all_df['Date']<all_dates[-test_cnt])]
    test_df = all_df[all_df['Date']>=all_dates[-test_cnt]]
    train_df['data_type'] = 'train'
    valid_df['data_type'] = 'valid'
    test_df['data_type'] = 'test'
    all_df = pd.concat([train_df, valid_df, test_df], axis=0)
    print('After splitting dataset：', train_df.shape, valid_df.shape, test_df.shape)
    return all_df


def get_horizontal_standard_feats(all_df):
    """
    Perform asset-based horizontal (time-along) standardization
    on features to address the inconsistent scaling of features 
    across different time series.

    Note: "asset-based" here refers to considering the features
    within each individual time series as assets and standardizing
    them accordingly, taking into account the variations within 
    each time series.
    """
    print('Standardlize horizontally:')
    con_feat_cols = ['Open','High','Low','Close','Adjusted_close']
    assets = all_df['asset'].unique()
    
    new_all_df = []
    for asset in tqdm(assets):
        try:
            a_df = all_df[all_df['asset']==asset]
            scaler = StandardScaler()
            scaler.fit(a_df[a_df['data_type']=='train'][con_feat_cols].values)
            stand_feat_arr = scaler.transform(a_df[con_feat_cols])
            stand_con_feat_cols = [f+'_Hnorm' for f in con_feat_cols]
            a_df[stand_con_feat_cols] = stand_feat_arr
            new_all_df.append(a_df)
        except:
            print('asset {} does not have enough length!'.format(asset))
    all_df = pd.concat(new_all_df, axis=0)
    return all_df


def get_vertical_standard_feats(all_df):
    """
    Perform time-based vertical (asset-along) standardization 
    on features to achieve relatively consistent scaling of 
    features across different time series, or meaningful 
    ranking of features (such as price, market capitalization,
     and rank).

    Note: "time-based" here refers to considering the variations
    of each feature across different time points within a single
    time series, and standardizing them accordingly based on the
    maximum and minimum values observed within that time series.
    """
    print('Standardlize vertically:')
    con_feat_cols = ['Adjusted_close','Volume','cash_flow','open_high_ratio','open_low_ratio','close_high_ratio','close_low_ratio','high_low_ratio']
    dates = all_df['Date'].unique()

    new_all_df = []
    for date in tqdm(dates):
        a_df = all_df[all_df['Date']==date]
        scaler = MinMaxScaler()
        stand_feat_arr = scaler.fit_transform(a_df[con_feat_cols].values)
        stand_con_feat_cols = [f+'_Vnorm' for f in con_feat_cols]
        a_df[stand_con_feat_cols] = stand_feat_arr
        new_all_df.append(a_df)
    all_df = pd.concat(new_all_df, axis=0)
    return all_df


def get_ts_stamp_feats(df):
    """Obtain Timestamp features"""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].apply(lambda x:x.year)
    df['Month'] = df['Date'].apply(lambda x:x.month)
    df['Weekofyear'] = df['Date'].apply(lambda x:x.weekofyear)
    df['Dayofyear'] = df['Date'].apply(lambda x:x.dayofyear)
    df['is_month_start'] = np.where(df['Date'].dt.is_month_start, 1, 0)
    df['is_month_end'] = np.where(df['Date'].dt.is_month_end, 1, 0)
    return df


def quantile02(x):
    return np.quantile(x, 0.2)

def quantile05(x):
    return np.quantile(x, 0.5)

def quantile08(x):
    return np.quantile(x, 0.8)

def skew(x):
    return x.skew()

def kurt(x):
    return x.kurt()


def agg_window_feats(all_df, feat_agg_dict, window_sizes=[22], window_weight=None):
    """Window-based statistical features (for individual assets)
      encompass different features, aggregation operations, and 
      aggregation window lengths."""
    new_all_df = []
    for window_size in window_sizes:
        print('Aggregate a window of features (size = %d):' % window_size)
        a_df_list = []
        # forloop each asset 
        for asset, a_df in tqdm(all_df.groupby('asset')):
            a_df = a_df.sort_values('Date', ascending=True)
            # Agg features for each asset
            for f in feat_agg_dict.keys():
                agg_df = a_df[f].rolling(window_size, window_weight).agg(feat_agg_dict[f])
                agg_df.columns=[f+'_win{}_'.format(window_size)+str(postfix) for postfix in list(agg_df)]
                a_df = pd.concat([a_df, agg_df], axis=1)
            a_df_list.append(a_df)
        window_all_df = pd.concat(a_df_list, axis=0)
        del a_df_list
        new_all_df.append(window_all_df)
    new_all_df = pd.concat(new_all_df, axis=1) 
    new_all_df = new_all_df.T.drop_duplicates().T
    return new_all_df


def agg_industry_feats(all_df, ind_agg_dict):
    """
    Industry aggregate statistical features (for multiple assets)
    encompass different features, aggregation operations, and 
    aggregation window lengths."""
    print('Aggregate an industry of features:')

    date_df_list = [] 
    for date, date_df in tqdm(all_df.groupby('Date')):
        new_ind_df_list = [] 
        # date_df.to_csv('./test.csv', index=False)
        for ind, ind_df in date_df.groupby('GICS_sector/ETF_type'):
            new_ind_df = ind_df.copy(deep=True)
            # print(ind_df.columns)
            ind_df_list = [] 
            for f in ind_agg_dict.keys():
                # if f not in ind_df.columns:
                #     print(f'{f} not in ind_df.columns')
                agg_df = new_ind_df[[f]].agg(ind_agg_dict[f])
                new_cols = [f+'_ind_'+str(postfix) for postfix in list(agg_df.index)]
                agg_df = agg_df.transpose()
                agg_df.columns=new_cols
                agg_df = agg_df.reset_index(drop=True)
                ind_df_list.append(agg_df)
            ind_agg_df = pd.concat(ind_df_list, axis=1)
            ind_agg_df['GICS_sector/ETF_type'] = ind
            new_ind_df = pd.merge(new_ind_df, ind_agg_df, how='left', left_on='GICS_sector/ETF_type', right_on='GICS_sector/ETF_type')
            new_ind_df_list.append(new_ind_df)
        date_df = pd.concat(new_ind_df_list, axis=0)
        date_df_list.append(date_df)

    new_all_df = pd.concat(date_df_list, axis=0)
    all_df = new_all_df.copy(deep=True)
    return all_df


def get_return_label(all_df):
    # Generating target prediction labels "return"
    print('Generate return label:')
    all_df_list = []
    for asset, a_df in tqdm(all_df.groupby('asset')):
        a_df = a_df.reset_index(drop=True)
        a_df['return'] = (a_df['Adjusted_close'].shift(-22) - a_df['Adjusted_close']) / a_df['Adjusted_close']
        all_df_list.append(a_df)
    all_df = pd.concat(all_df_list, axis=0).reset_index(drop=True)
    return all_df


def extract_seasonal_feat(all_df, f, window_size):
    """Extract seasonal features"""
    all_df.reset_index(drop=True, inplace=True)
    all_df[f+'_season'] = all_df[f] - all_df[f + '_win{}_mean'.format(window_size)]
    return all_df


def symbol2assetID(meta_df, data_df):
    """Transform asset symbol to asset ID 
    inputs：
        meta_df     (pd.DataFrame): M6_Universe.csv dataframe
        data_df     (pd.DataFrame): dateframe with asset symbol
    returns：
        data_df     (pd.DataFrame): dataframe with asset id
    """
    #label encoder
    assets = meta_df['symbol'].unique() 
    asset2id_dict = {assets[i]:i for i in range(len(assets))}
    data_df['symbol'] = data_df['asset'].values
    data_df['asset'] = data_df['asset'].apply(lambda x:asset2id_dict[x])   
    return data_df


def add_type_id(meta_df, df):
    """Add asset's type_id and subtype_id"""
    meta_df['asset'] = meta_df['symbol'].values

    types = meta_df['GICS_sector/ETF_type'].unique() 
    type2id_dict = {types[i]:i for i in range(len(types))}
    meta_df['type_id'] = meta_df['GICS_sector/ETF_type'].apply(lambda x:type2id_dict[x])

    subtypes = meta_df['GICS_industry/ETF_subtype'].unique() 
    subtype2id_dict = {subtypes[i]:i for i in range(len(subtypes))}
    meta_df['subtype_id'] = meta_df['GICS_industry/ETF_subtype'].apply(lambda x:subtype2id_dict[x])

    new_df = pd.merge(left=df, right=meta_df[['asset', 'type_id', 'subtype_id']], how='left', left_on=['asset'], right_on=['asset'])
    return new_df


def get_return_rank_label2(df):
    """Return data with soft label(Rank1/2/3/4/5) and hard label(Rank)
    The difference with get_return_rank_label() is：
        disadvantage: W/O Individual handling for cases where the asset 
                      quantity is too low under a single date or when 
                      the rank is at the boundary of a soft distribution
                      allocation.
        advantage：The label generation speed is fast and there are no 
                   abnormal labels caused by unconsidered boundary
                   conditions.
    Use Case：all_df = utils.get_return_rank_label2(all_df)
    """
    rank_array = np.zeros((len(df), 6))
    dates = df['Date'].unique()
    for date in dates:
        daily_returns = df[df['Date']==date][['asset','return']]
        daily_returns_index = daily_returns.index
        daily_returns = daily_returns.fillna(0) # 新加的
        data = rankdata(daily_returns['return'])
        data = np.array(data).astype(int) // 20.01 + 1
        data = data.astype(int)
        rank_data = np.eye(5)[data-1]
        rank_data = pd.DataFrame(rank_data, columns=['Rank1','Rank2','Rank3','Rank4','Rank5'])
        rank_data['Rank'] = np.argmax(rank_data[['Rank1','Rank2','Rank3','Rank4','Rank5']].values, axis=1)
        rank_array[daily_returns_index] = rank_data[['Rank','Rank1','Rank2','Rank3','Rank4','Rank5']]
    df[['Rank','Rank1','Rank2','Rank3','Rank4','Rank5']] = rank_array
    return df


def run_feature_enginner(data_path, meta_df, train_ratio, test_cnt, feat_agg_dict, window_sizes, window_weight, ind_agg_dict):
    """Run feature enginnering"""
    all_df = load_all_files(data_path)
    all_df = get_basic_feats(all_df)
    all_df = drop_date_with_less_asset(all_df, asset_num=80)
    all_df = add_data_type(all_df, train_ratio, test_cnt)
    all_df = get_horizontal_standard_feats(all_df)
    all_df = get_vertical_standard_feats(all_df)
    all_df = agg_window_feats(all_df, feat_agg_dict, window_sizes, window_weight)

    # Due to the sliding window statistics performed earlier,
    # there may be missing values generated within the initial
    # time period where statistics cannot be calculated. 
    # These sample points need to be removed.
    all_df.drop(index = all_df[all_df['Adjusted_close_Hnorm_win22_mean'].isnull()].index, axis=0, inplace=True)
    
    # @ahf：The introduction of industry information does not bring improvement to the model.
    all_df = pd.merge(all_df, meta_df[['symbol','GICS_sector/ETF_type']], how='left', left_on='asset', right_on='symbol')
    all_df = agg_industry_feats(all_df, ind_agg_dict)
    

    # all_df = extract_seasonal_feat(all_df, f='Adjusted_close_Hnorm', window_size=22)
    all_df = get_return_label(all_df)
    all_df = add_type_id(meta_df, all_df)
    all_df = symbol2assetID(meta_df, all_df)
    all_df = get_return_rank_label2(all_df)

    # Obtain the predict.csv file, which contains the latest
    # sample for each asset based on its most recent date.
    null_return_df = all_df[all_df['return'].isnull()]
    predict_df_list = []
    for asset, a_df in null_return_df.groupby('asset'):
        predict_df_list.append(a_df.iloc[[-1],:])
    predict_df = pd.concat(predict_df_list, axis=0).reset_index(drop=True)

    # gain train/valid/test.csv
    train_df = all_df[all_df['data_type']=='train'].reset_index(drop=True)
    valid_df = all_df[all_df['data_type']=='valid'].reset_index(drop=True)
    test_df = all_df[all_df['data_type']=='test'].reset_index(drop=True)

    # save dataset
    feat_type = 'BF_TF_ARF_WF'
    train_df.to_csv(f'{self_data_path}train_rank_df_{feat_type}.csv', index=False)
    valid_df.to_csv(f'{self_data_path}valid_rank_df_{feat_type}.csv', index=False)
    test_df.to_csv(f'{self_data_path}test_rank_df_{feat_type}.csv', index=False)
    predict_df.to_csv(f'{self_data_path}predict_rank_df_{feat_type}.csv', index=False)

    print("Finish Feature Engineering!")
    print('train:{}, valid:{}, test:{}, predict:{}'.format(train_df.shape, valid_df.shape, test_df.shape, predict_df.shape))
    return train_df, valid_df, test_df, predict_df


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='M6 data_crawler')
    parser.add_argument('--data_path', type=str, required=True, default="root_path/data/", help='crawled data saving path')
    parser.add_argument('--self_data_path', type=str, required=True, default="root_path/pp_data/", help='M6_Universe data path')
    args = parser.parse_args()

    data_path = args.data_path
    self_data_path = args.self_data_path
    meta_path = self_data_path + 'M6_Universe.csv'
    meta_df = pd.read_csv(meta_path)

    # Window-agg features
    feat_agg_dict = {
    # Horizontal standardized feature aggregation engineering
    'Open_Hnorm':[np.mean, np.max, np.min, np.var, quantile02, quantile05, quantile08, kurt, skew],
    'High_Hnorm':[np.mean, np.max, np.min, np.var, quantile02, quantile05, quantile08, kurt, skew],
    'Low_Hnorm':[np.mean, np.max, np.min, np.var, quantile02, quantile05, quantile08, kurt, skew],
    'Close_Hnorm':[np.mean, np.max, np.min, np.var, quantile02, quantile05, quantile08, kurt, skew],
    'Adjusted_close_Hnorm':[np.mean, np.max, np.min, np.var, quantile02, quantile05, quantile08, kurt, skew],
    # Vertical standardized feature aggregation engineering
    'Adjusted_close_Vnorm':[np.mean, np.max, np.min, np.var, quantile02, quantile05, quantile08, kurt, skew],
    'Volume_Vnorm':[np.mean, np.max, np.min, np.var, quantile02, quantile05, quantile08, kurt, skew],
    'cash_flow_Vnorm':[np.mean, np.max, np.min, np.var, quantile02, quantile05, quantile08, kurt, skew],
    'open_high_ratio_Vnorm':[np.mean, np.max, np.min, np.var, quantile02, quantile05, quantile08, kurt, skew],
    'open_low_ratio_Vnorm':[np.mean, np.max, np.min, np.var, quantile02, quantile05, quantile08, kurt, skew],
    'close_high_ratio_Vnorm':[np.mean, np.max, np.min, np.var, quantile02, quantile05, quantile08, kurt, skew],
    'close_low_ratio_Vnorm':[np.mean, np.max, np.min, np.var, quantile02, quantile05, quantile08, kurt, skew],
    'high_low_ratio_Vnorm':[np.mean, np.max, np.min, np.var, quantile02, quantile05, quantile08, kurt, skew],
    }
    window_sizes = [22]
    window_weight = None # Support scipy window-weighted funcs：https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows

    # # industry-agg features
    # # Note: if each class only has a single value, using var and other agg function will lead to None
    ind_agg_dict = {
    'Adjusted_close_Vnorm_win22_mean':[np.mean],
    'Volume_Vnorm_win22_mean':[np.mean],
    'cash_flow_Vnorm_win22_mean':[np.mean],
    'open_high_ratio_Vnorm_win22_mean':[np.mean],
    'open_low_ratio_Vnorm_win22_mean':[np.mean],
    'close_high_ratio_Vnorm_win22_mean':[np.mean],
    'close_low_ratio_Vnorm_win22_mean':[np.mean],
    'high_low_ratio_Vnorm_win22_mean':[np.mean]
    }

    train_ratio = 0.7 
    test_cnt = 160    

    # Run feature engineering
    train_df, valid_df, test_df, predict_df = run_feature_enginner(data_path, meta_df,
                                                                    train_ratio, test_cnt,
                                                                    feat_agg_dict, window_sizes, window_weight, ind_agg_dict)
