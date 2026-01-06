import numpy as np
import yfinance as yf
import NNTrader as nt
import torch as pt
import datetime as dt
import indicators as inds
import pandas as pd
import matplotlib.pyplot as plt

def get_inds(df):
    # Each function takes a dataframe with a 'Close' (and 'Volume' column in the case of mfi) and adds a column with the specified indicator
    df = inds.percentage_price(df)
    df = inds.vol(df)
    df = inds.momentum(df)
    df = inds.aroon(df)
    df = inds.mfi(df).drop(columns=['Close', 'Volume']).fillna(0)
    return df

position_size = 10000
# tickers = ['MSFT', 'XOM', 'GLD', 'T', 'JPM', 'IT']
tickers = ['SPY']

backdate = dt.datetime(year=2019, month=11, day=1)
sd = dt.datetime(year=2020, month=1, day=1)
val_start = dt.datetime(year=2023, month=1, day=1)
test_start = dt.datetime(year=2024, month=1, day=2)

for ticker in tickers:
    data = yf.Ticker(ticker).history(start=backdate, interval='1d').drop(columns=['High', 'Low', 'Open', 'Dividends', 'Stock Splits'])
    data.index = data.index.tz_localize(None)


    test_data = data.loc[test_start:, 'Close']
    benchmark = (test_data.iloc[-1] / test_data.iloc[0] - 1) * position_size

    w_size = 5
    dr = data['Close'] / data['Close'].iloc[0]
    dr[1:] = (dr[1:] / dr[:-1].values) - 1
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=w_size)
    y = dr.rolling(window=indexer, min_periods=1).sum()

    # print(y)

    data = get_inds(data)

    X_train = pt.tensor(data[sd:val_start].values, dtype=pt.float32)
    X_val = pt.tensor(data[val_start:test_start].values, dtype=pt.float32)
    X_test = pt.tensor(data[test_start:].values, dtype=pt.float32)

    y_train = pt.from_numpy(y[sd:val_start].values.astype(np.float32).reshape(-1,1))
    y_val = pt.from_numpy(y[val_start:test_start].values.astype(np.float32).reshape(-1,1))
    y_test = pt.from_numpy(y[test_start:].values.astype(np.float32).reshape(-1,1))

    runs = 100
    rets = np.zeros(runs)


    for run in range(runs):

        model = nt.NNTrader(n_features=X_train.shape[1])

        model.nn_train(X_train, y_train, X_val, y_val)

        y_pred_test, y_test_new = model.nn_test(X_test, y_test)

        avg = np.mean(np.sign(y_pred_test) == np.sign(y_test_new))

        # print('Sign Mean:', avg)

        theta = np.std(y_pred_test) * 0.25
        positions = np.where(
            y_pred_test >  theta,  1,
            np.where(y_pred_test < -theta, -1, 0)
        )

        returns = dr[test_start:] * position_size
        cost = 0.0005

        turnover = np.abs(np.diff(positions, prepend=0))
        pnl = positions * returns - cost * turnover

        cum_ret = np.cumsum(pnl)
        rets[run] = cum_ret.iloc[-1]

    # print(positions)

    # plt.plot(pnl)
    # plt.show()
    # plt.close()
    print('| ', ticker, ' | ', np.mean(rets), ' | ', benchmark, ' |')

    # print('Average Return for:', np.mean(rets))
    # print('Benchmark Return:', benchmark)


    # plt.plot(cum_ret)
    # plt.show()
    # plt.close()



