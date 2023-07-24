"""
data_crawler.py
@author: aihongfeng
@date: 20220413
@function: Crawl all assets' EOD basic data
"""
import argparse
import pandas as pd
from eod_historical_data import get_eod_data
import datetime
from tqdm import tqdm


def crawl_eod_data(meta_path='root_path/data/M6_Universe.csv',
                   api_key="624**********",
                   save_path="root_path/m6_data/", year_duration=5):
    """Crawl all assets' EOD basic data
    Ref EOD API Doc：https://github.com/femtotrader/python-eodhistoricaldata"""
   # Gain all assets
    meta_df = pd.read_csv(meta_path)
    assets = meta_df.symbol.unique() 

    # Crawl data
    end_time = pd.to_datetime(datetime.date.today())
    start_time = end_time - datetime.timedelta(365*year_duration)
    print('Crawling Period from {} to {}, Total duration: {} years.'.format(start_time.date(), end_time.date(), year_duration))

    fail_assets = []
    for asset in tqdm(assets):
        try:
            if asset in ['SEGA.L','IEAA.L','HIGH.L','JPEA.L','IUVL.L','IUMO.L','SPMV.L','IEVL.L','IEFM.L','MVEU.L']:
                exchange = "LSE" # London
                df = get_eod_data(asset.split('.')[0], exchange, api_key=api_key, start=start_time, end=end_time)
            else:
                exchange = "US" # USA
                df = get_eod_data(asset, exchange, api_key=api_key, start=start_time, end=end_time)
            df = df.reset_index()
            df.to_csv(save_path + '{}.csv'.format(asset), index=False)
        except:
            fail_assets.append(asset)
    print("Fail Crawling with: ", fail_assets)


if __name__=="__main__":
    # Crawl all assets' EOD basic data with the crawling year length
    parser = argparse.ArgumentParser(description='M6 data_crawler')
    parser.add_argument('--meta_path', type=str, required=True, default='root_path/eod_data/M6_Universe.csv', help='meta data path')
    parser.add_argument('--api_key', type=str, required=True, default="624**********", help='EOD api key')
    parser.add_argument('--save_path', type=str, required=True, default="root_path/data/", help='crawled data saving path')
    parser.add_argument('--year_duration', type=int, required=True, default=5, help='the crawling year length')
    args = parser.parse_args()

    crawl_eod_data(meta_path=args.meta_path,
                    api_key=args.api_key,
                    save_path=args.save_path,
                    year_duration=args.year_duration)


