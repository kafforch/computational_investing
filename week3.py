# QSTK Imports
from math import sqrt

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def simulate(startdate, enddate, stocklist, allocations):

    assert sum(allocations) == 1.0, "Allocations have to add up to 1.0"
    assert len(stocklist) == len(allocations), "Allocations number has to match number of stocks"

    # Create a data frame with stock symbols
    c_dataobj = da.DataAccess('Yahoo')
    dtlist_end = map(lambda x: int(x), enddate.split("-"))
    dt_end = dt.datetime(year=dtlist_end[0], month=dtlist_end[1], day=dtlist_end[2])
    dtlist_start = map(lambda x: int(x), startdate.split("-"))
    dt_start = dt.datetime(year=dtlist_start[0], month=dtlist_start[1], day=dtlist_start[2])
    dt_timeofday = dt.timedelta(hours=16)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, stocklist, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))


    # Copying close price into separate dataframe to find rets
    df_rets = d_data['close'].copy()
    # Filling the data.
    df_rets = df_rets.fillna(method='ffill')
    df_rets = df_rets.fillna(method='bfill')
    df_rets = df_rets.fillna(1.0)

    # Numpy matrix of filled data values
    na_rets = df_rets.values

    na_norm_price = na_rets / na_rets[0, :]

    # portfolio value = sum of normalized prices * Allocation ratio
    na_port_value = np.sum( na_norm_price * allocations, axis=1)

    na_portrets = na_port_value.copy()
    tsu.returnize0(na_portrets)

    # Average return
    daily_ret = np.average(na_portrets)

    # Estimate portfolio returns
    cum_ret = na_port_value[-1]/na_port_value[0]

    # Volatility - std dev
    vol = np.std(na_portrets)

    # Sharpe
    sharpe = sqrt(252) * daily_ret / vol

    return vol, daily_ret, sharpe, cum_ret

def main():
    #print simulate('2011-01-01', '2011-12-31', ['AAPL', 'GLD', 'GOOG', 'XOM'], [0.4, 0.4, 0.0, 0.2])
    #print simulate('2010-01-01', '2010-12-31', ['AXP', 'HPQ', 'IBM', 'HNZ'], [0.0, 0.0, 0.0, 1.0])

    combinations = np.arange(start=0, stop=10000)
    indices = []

    def conv(number, pos):
        try:
            return int((str(number))[pos]) / 10.
        except IndexError:
            return 0.

    def plot_winner(startdate, enddate, symbols, alloc):
        # Create a data frame with stock symbols
        c_dataobj = da.DataAccess('Yahoo')
        dtlist_end = map(lambda x: int(x), enddate.split("-"))
        dt_end = dt.datetime(year=dtlist_end[0], month=dtlist_end[1], day=dtlist_end[2])
        dtlist_start = map(lambda x: int(x), startdate.split("-"))
        dt_start = dt.datetime(year=dtlist_start[0], month=dtlist_start[1], day=dtlist_start[2])
        dt_timeofday = dt.timedelta(hours=16)
        ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

        # Keys to be read from the data, it is good to read everything in one go.
        ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

        # Reading the data, now d_data is a dictionary with the keys above.
        # Timestamps and symbols are the ones that were specified before.
        ldf_data = c_dataobj.get_data(ldt_timestamps, symbols, ls_keys)
        ldf_spy_data = c_dataobj.get_data(ldt_timestamps, ['SPY'], ['close'])
        d_data = dict(zip(ls_keys, ldf_data))
        d_spy_data = dict(zip(["close"], ldf_spy_data))

        # Copying close price into separate dataframe to find rets
        df_rets = d_data['close'].copy()
        # Filling the data.
        df_rets = df_rets.fillna(method='ffill')
        df_rets = df_rets.fillna(method='bfill')
        df_rets = df_rets.fillna(1.0)

        # Numpy matrix of filled data values
        na_rets = df_rets.values
        d_spy = d_spy_data['close'].copy()
        df_spy = d_spy.copy()
        na_spy = np.sum(df_spy.values, axis=1)

        na_norm_spy = na_spy / na_spy[0]

        na_norm_price = na_rets / na_rets[0, :]

        # portfolio value = sum of normalized prices * Allocation ratio
        na_port_value = np.sum( na_norm_price * alloc, axis=1)

        plt.clf()
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(ldt_timestamps, na_port_value)
        plt.plot(ldt_timestamps, na_norm_spy)
        label = ", ".join(
                map(lambda x: x[0] + " " + str(x[1]), zip(symbols, alloc))
        )
        ls_names = [label]
        ls_names.append('SPY')
        plt.legend(ls_names, loc="lower left", fontsize="small")
        plt.ylabel('Cumulative Returns')
        plt.xlabel('Date')
        fig.autofmt_xdate(rotation=45)
        plt.savefig('mytutorial3.pdf', format='pdf')

    for num in combinations:
        num_0 = conv(num,0)
        num_1 = conv(num,1)
        num_2 = conv(num,2)
        num_3 = conv(num,3)
        if (num_0 + num_1 + num_2 + num_3) == 1.:
            indices.append([num_0, num_1, num_2, num_3])

    startdate, enddate, portfolio = '2011-01-01', '2011-12-31', ['AAPL', 'GLD', 'GOOG', 'XOM']

    highest_sharpe = 0
    best_alloc = []
    for alloc in indices:
        vol, daily_ret, sharpe, cum_ret = simulate(startdate, enddate, portfolio, alloc)
        if sharpe > highest_sharpe:
            highest_sharpe = sharpe
            best_alloc = alloc

    print highest_sharpe
    print best_alloc
    plot_winner(startdate, enddate, portfolio, best_alloc)


if __name__ == '__main__':
    main()