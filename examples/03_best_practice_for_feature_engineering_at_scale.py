#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import featurizer.functors.talib as talib
import featurizer.functors.volume_price as vp
import featurizer.functors.journalhub as jf
import featurizer.functors.time_series as tf

# ============================================================ #
# step1: define your custom featurizer                         #
# ============================================================ #

class DefaultFeaturizer(object):
    
    def __init__(self):
        self.pct_change = tf.PctChange(window=1) 
        
        # price-volume
        self.PriceVolume = talib.PriceVolume()
        
        
        # extra added
        self.KDJ = talib.KDJRelated(fastk_period=9, slowk_period=3, slowd_period=3)
        
        # journal
        self.ReturnsRollingStd4 = jf.ReturnsRollingStd(window=4)
        self.ReturnsRollingStd12 = jf.ReturnsRollingStd(window=12)
        
        self.BackwardSharpRatio4 = jf.BackwardSharpRatio(window=4)
        self.BackwardSharpRatio12 = jf.BackwardSharpRatio(window=12)
        
        # trading factor
        self.VolumeReturnsCorr4 = vp.VolumeReturnsCorr(window=4)
        self.VolumeReturnsCorr12 = vp.VolumeReturnsCorr(window=12)
        
        self.HighLowCorr4 = vp.HighLowCorr(window=4)
        
        
    def forward(self, open_ts, high_ts, low_ts, close_ts, volume_ts):
        feature_list = []
        feature_name_list = []
        # data
        returns_ts = self.pct_change(close_ts)
        # 4
        PriceVolume = self.PriceVolume(close_ts, volume_ts)
        feature_list.extend([PriceVolume])
        feature_name_list.extend(["PriceVolume"])
        # dkj
        RSV, K, D, J = self.KDJ(high_ts, low_ts, close_ts)
        feature_list.extend([RSV, K, D, J])
        feature_name_list.extend("RSV, K, D, J".split(","))
        # journalhub
        ReturnsRollingStd4 = self.ReturnsRollingStd4(returns_ts)
        ReturnsRollingStd12 = self.ReturnsRollingStd12(returns_ts)
        
        BackwardSharpRatio4 = self.BackwardSharpRatio4(returns_ts)
        BackwardSharpRatio12 = self.BackwardSharpRatio12(returns_ts)
        
        feature_list.extend([ReturnsRollingStd4,ReturnsRollingStd12,BackwardSharpRatio4,BackwardSharpRatio12])
        feature_name_list.extend("ReturnsRollingStd4,ReturnsRollingStd12,BackwardSharpRatio4,BackwardSharpRatio12".split(","))
        #
        
        VolumeReturnsCorr4 = self.VolumeReturnsCorr4(volume_ts, returns_ts)
        VolumeReturnsCorr12 = self.VolumeReturnsCorr12(volume_ts, returns_ts)
        
        HighLowCorr4 = self.HighLowCorr4(high_ts, low_ts)
        
        feature_list.extend([VolumeReturnsCorr4,VolumeReturnsCorr12,HighLowCorr4])
        feature_name_list.extend("VolumeReturnsCorr4,VolumeReturnsCorr12,HighLowCorr4".split(","))
        
        # label
        feature_list.extend([returns_ts])
        feature_name_list.extend(["returns"])
        return feature_list, feature_name_list


# ======================================================================= #
# step2: get data                                                         #
# ======================================================================= #
import os
import jqdatasdk
from xqdata.api import history_bars

jqdata_username = os.environ["JQDATA_USERNAME"]
jqdata_password = os.environ["JQDATA_PASSWORD"]
jqdatasdk.auth(username=jqdata_username, password=jqdata_password)


order_book_ids = ['600000.XSHG',"601336.XSHG","600570.XSHG",'000001.XSHE',"300015.XSHE"]

all_fields = ["open", "high", "low","close","volume"]
bar_count=500
dt = "2019-08-20"
frequency="1d"
data_df = history_bars(order_book_ids=order_book_ids, bar_count=bar_count, frequency=frequency, fields=all_fields, dt=dt, skip_suspended=False)
# raw data fillna
data_df = data_df.groupby(by="order_book_id").apply(lambda x:x.fillna(method="ffill"))

# =================================================================== #
# step3: create feature                                               #
# =================================================================== #
import torch
import pandas as pd

def create_raw_feature(raw_data: pd.DataFrame) -> pd.DataFrame :  

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    open_ts = torch.tensor(raw_data["open"].unstack(0).values, dtype=torch.float32, device=device)
    high_ts = torch.tensor(raw_data["high"].unstack(0).values, dtype=torch.float32, device=device)
    low_ts = torch.tensor(raw_data["low"].unstack(0).values, dtype=torch.float32, device=device)
    close_ts = torch.tensor(raw_data["close"].unstack(0).values, dtype=torch.float32, device=device)
    volume_ts = torch.tensor(raw_data["volume"].unstack(0).values, dtype=torch.float32, device=device)
    
    featurizer = DefaultFeaturizer()
    feature_list, feature_name_list = featurizer.forward(open_ts, high_ts, low_ts, close_ts, volume_ts)
    #pdb.set_trace()
    data_container = {}
    for i, feature in enumerate(feature_list):
        raw_feature_df = pd.DataFrame(feature.cpu().numpy(), index=raw_data.index.levels[1], columns=raw_data.index.levels[0])
        data_container[feature_name_list[i]] = raw_feature_df
            
    featured_df = pd.concat(data_container)
    featured_df = featured_df.stack(0).unstack(0).swaplevel(0,1).sort_index(level=0)
    featured_df.rename_axis(index=["order_book_id", "datetime"])
    return featured_df[feature_name_list] 

feature_df = create_raw_feature(data_df)

