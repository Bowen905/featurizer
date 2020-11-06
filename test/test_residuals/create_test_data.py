# -*- coding: utf-8 -*-

# Author: Huanqiu Wang
# Created on: Nov 6, 2020

'''
This module contains the code that creates data for test cases, including
    - data needed while running the unit tests
    - the expected input data
    - the expected output data to be compared with the dynamic outputs
    
Note: If you want to re-run this file to create a different set of data, please make sure 
      that the functions and functors involved are correct, so that the resulting datasets
      are not erroneous themselves

'''

import pandas as pd
import pickle
import torch

############### get stocks and index dataframes from jointquant, then store to pickle
from jqdatasdk import *
auth('17727977512', '147258369')

stocks = ['000001.XSHE', '000002.XSHE']
index = ['399006.XSHE']
fields=['date','open','high','low','close','volume','money']

stocks_100d = get_bars(stocks, fields= fields, unit= '1d', end_dt="2020-11-06", count=100)
stocks_100d = stocks_100d.reset_index(level=1, drop=True).set_index("date", append=True)
stocks_100d.index.rename(['order_book_id','datetime'], inplace=True)
stocks_100d.rename(columns= {'money': 'total_turnover'}, inplace=True)
stocks_100d = stocks_100d.groupby(by="order_book_id").apply(lambda x:x.fillna(method="ffill"))

index_100d = get_bars(index, fields= fields, unit= '1d', end_dt="2020-11-06", count=100)
index_100d = index_100d.reset_index(level=1, drop=True).set_index("date", append=True)
index_100d.index.rename(['order_book_id','datetime'], inplace=True)
index_100d.rename(columns= {'money': 'total_turnover'}, inplace=True)
index_100d = index_100d.groupby(by="order_book_id").apply(lambda x:x.fillna(method="ffill"))

stocks_100d.to_pickle('stocks_100d.pkl')
index_100d.to_pickle('index_100d.pkl')


############## get input x and y tensors
import featurizer.functors.journalhub as jf
import get_securities_fields_tensors as gt
import featurizer.functors.time_series as tf

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

dict_index = gt.get_securities_fields_tensors(index_100d)
dict_stocks = gt.get_securities_fields_tensors(stocks_100d)

stocks_close_ts = dict_stocks['close']
stocks_high_ts = dict_stocks['high']
stocks_low_ts = dict_stocks['low']
stocks_turnover_ts = dict_stocks['turnover']
stocks_volume_ts = dict_stocks['volume']
index_close_ts = dict_index['close']
index_volume_ts = dict_index['volume']
index_turnover_ts = dict_index['turnover']

pct_change_functor = tf.PctChange()
index_returns_ts = pct_change_functor(index_close_ts)
stocks_returns_ts = pct_change_functor(stocks_close_ts)

index_returns_ts_temp = index_returns_ts
for i in range(0, list(stocks_returns_ts.size())[1] - 1):
    index_returns_ts = torch.cat((index_returns_ts, index_returns_ts_temp), 1)
    
stock_turnover_pct_ts = pct_change_functor(stocks_turnover_ts)

stock_high_low_ratio_ts = torch.div(stocks_high_ts, stocks_low_ts)
# --- stack and permute tensors
x_ts = torch.stack([stock_turnover_pct_ts, stock_high_low_ratio_ts, index_returns_ts])
x_ts = x_ts.permute(2,1,0)
y_ts = stocks_returns_ts.unsqueeze(-1).permute(1,0,2)