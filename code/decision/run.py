
import numpy as np
import pandas as pd
import os
from scipy.stats import rankdata
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from utils import optimal_portfolio, DE_portfolio, black_litterman


class Test:
    def __init__(self):
        self.freq = '1min'
        return

    # def preprocess_data(self):
    #     def read_history(file):
    #         df = pd.read_csv(file, header=0)
    #         df['Date'] = pd.to_datetime(df['Date'])
    #         df.rename(columns={'Adjusted_close':'Adj Close'}, inplace=True)
    #         return df
    #     data_dir = '../../data'
    #     files = os.listdir(data_dir)
    #     data = []
    #     for file in files:
    #         stock = file.replace('.csv', '')
    #         file = os.path.join(data_dir, file)
    #         df = read_history(file)
    #         df['stock'] = stock.strip()
    #         data.append(df)

    #     data = pd.concat(data)

    #     # Fill in the missing dates
    #     dates = data[['Date']].drop_duplicates()
    #     dates['x'] = 1
    #     stocks = data[['stock']].drop_duplicates()
    #     stocks['x'] = 1
    #     ds = dates.merge(stocks)
    #     data = ds.merge(data, how='left').drop('x', axis=1).sort_values(['stock', 'Date']).fillna(method='ffill')

    #     ### add label
    #     trading_dates = data['Date'].values
    #     dates = pd.date_range(trading_dates.min(), trading_dates.max())
    #     dates = pd.DataFrame(dates, columns=['date'])
    #     dates['isin'] = dates['date'].isin(trading_dates)
    #     dates['last_trading_date'] = dates['date']
    #     dates.loc[~dates['isin'], 'last_trading_date'] = None
    #     dates['last_trading_date'] = dates['last_trading_date'].fillna(method='ffill')

    #     last_trading_date_dic = dict(zip(dates['date'], dates['last_trading_date']))

    #     data['date_4w'] = data['Date'] + pd.Timedelta(days=28)  
    #     data['date_4w'] = data['date_4w'].apply(lambda x:last_trading_date_dic.get(x, None)) # # 4周后上一个交易日
    #     next_prices = data[['Date', 'stock', 'Adj Close']]
    #     next_prices.columns = ['date_4w', 'stock', 'next_price']
    #     data = data.merge(next_prices, how='left')


    #     returns = {}
    #     for stock in data.stock.unique():
    #         tmp = data[data.stock == stock]
    #         tmp['next Close'] = tmp['Adj Close'].diff()#.iloc[1:].reset_index(drop=True)
    #         tmp = tmp.drop_duplicates(subset='Date', keep='first')
    #         return_single = tmp['next Close'] / tmp['Adj Close'].shift(1)
    #         if len(return_single) > 1287:
    #             print(stock)
    #         returns[stock] = return_single.iloc[1:].tolist()

    #     pd.DataFrame(returns).to_csv('real_returns_0501.csv', index=False)
    #     return returns


    def run(self):
        return_vec = pd.read_csv('real_returns_0501.csv')
        # print(return_vec.shape)
        black_litterman(return_vec)
        # weights, returns, risks = optimal_portfolio(return_vec.to_numpy())

        # de = DE_portfolio(constraint_eq=[], 
        #                     constraint_ueq=[
        #                                     # 小于等于0
        #                                     lambda x: sum([abs(i) for i in x]) - 1,
        #                                     lambda x: 0.25 - sum([abs(i) for i in x])
        #                                 ])

        # vals, weights = de.run()


# if __name__ == "__main__":
#     t = Test()
#     t.run()
