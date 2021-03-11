# -*- coding: utf-8 -*-
# Copyright StateOfTheArt.quant. 
#
# * Commercial Usage: please contact allen.across@gmail.com
# * Non-Commercial Usage:
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import pdb
import torch
import pandas as pd
import numpy as np
import featurizer.functions.time_series_functions as tsf

def macd(tensor, fastperiod=12, slowperiod=26, signalperiod=9):
    #DIF = tsf.ema(tensor, fastperiod) - tsf.ema(tensor, slowperiod)
    #DEA = tsf.ema(DIF, signalperiod)
    #MACD = (DIF - DEA) * 1 # Here is 1 rather than trodational 2
    import talib
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    DIF = tensor_df.apply(lambda x: talib.MACD(x, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[0])
    DEA = tensor_df.apply(lambda x: talib.MACD(x, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[1])
    MACD = tensor_df.apply(lambda x: talib.MACD(x, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[2])
    
    DIF_ts = torch.tensor(DIF.values, dtype=tensor.dtype, device=tensor.device)
    DEA_ts = torch.tensor(DEA.values, dtype=tensor.dtype, device=tensor.device)
    MACD_ts = torch.tensor(MACD.values, dtype=tensor.dtype, device=tensor.device)
    #
    return DIF_ts, DEA_ts, MACD_ts

    return DIF, DEA, MACD

def macd_(tensor, fastperiod=12, slowperiod=26, signalperiod=9):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    datas = pd.Series(tensor_df[0].values)
    DIF = [0]*slowperiod
    for i in range(slowperiod, len(tensor_np)):
        dif = tsf.ema_(torch.tensor(datas[i-fastperiod:i].values), fastperiod) - tsf.ema_(torch.tensor(datas[i-slowperiod:i].values), slowperiod)
        DIF.append(dif)
    
    DEA = [0]*(slowperiod+signalperiod)
    MACD = [np.nan]*(slowperiod+signalperiod)
    for i in range(slowperiod+signalperiod, len(tensor_np)):
        dea = tsf.ema_(torch.tensor(DIF[i-signalperiod:i]), signalperiod)
        DEA.append(dea)
        macd = 2*(DIF[i]-DEA[i])
        MACD.append(macd)
    MACD_ts = torch.tensor(MACD, dtype=tensor.dtype, device=tensor.device)
    return MACD_ts
        

def macdext(close_ts, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    DIF = close_df.apply(lambda x: talib.MACDEXT(x, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)[0])
    DEA = close_df.apply(lambda x: talib.MACDEXT(x, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)[1])
    MACD = close_df.apply(lambda x: talib.MACDEXT(x, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)[2])
    
    DIF_ts = torch.tensor(DIF.values, dtype=close_ts.dtype, device=close_ts.device)
    DEA_ts = torch.tensor(DEA.values, dtype=close_ts.dtype, device=close_ts.device)
    MACD_ts = torch.tensor(MACD.values, dtype=close_ts.dtype, device=close_ts.device)
    
    return DIF_ts, DEA_ts, MACD_ts


def macdfix(close_ts, signalperiod=9): # fixed fastperiod=12 and slowperiod=26
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    DIF = close_df.apply(lambda x: talib.MACDFIX(x, signalperiod=9)[0])
    DEA = close_df.apply(lambda x: talib.MACDFIX(x, signalperiod=9)[1])
    MACD = close_df.apply(lambda x: talib.MACDFIX(x, signalperiod=9)[2])
    
    DIF_ts = torch.tensor(DIF.values, dtype=close_ts.dtype, device=close_ts.device)
    DEA_ts = torch.tensor(DEA.values, dtype=close_ts.dtype, device=close_ts.device)
    MACD_ts = torch.tensor(MACD.values, dtype=close_ts.dtype, device=close_ts.device)
    
    return DIF_ts, DEA_ts, MACD_ts


def ppo(close_ts, fastperiod=12, slowperiod=26, matype=0):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    PPO = close_df.apply(lambda x: talib.PPO(x, fastperiod=12, slowperiod=26, matype=0))
    PPO_ts = torch.tensor(PPO.values, dtype=close_ts.dtype, device=close_ts.device)
    return PPO_ts


def rsi(tensor, timeperiod):
    import talib
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    tensor_df = tensor_df.apply(lambda x: talib.RSI(x, timeperiod=timeperiod))
    output_tensor = torch.tensor(tensor_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor


def bbands(tensor, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
    import talib
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    upperband = tensor_df.apply(lambda x: talib.BBANDS(x, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[0])
    middleband = tensor_df.apply(lambda x: talib.BBANDS(x, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[1])
    lowerband = tensor_df.apply(lambda x: talib.BBANDS(x, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[2])
    
    upperband_ts = torch.tensor(upperband.values, dtype=tensor.dtype, device=tensor.device)
    middleband_ts = torch.tensor(middleband.values, dtype=tensor.dtype, device=tensor.device)
    lowerband_ts = torch.tensor(lowerband.values, dtype=tensor.dtype, device=tensor.device)
    return upperband_ts, middleband_ts, lowerband_ts   

def kdj(high_ts, low_ts, close_ts, fastk_period=9, slowk_period=3, slowd_period=3):
    range_ts = tsf.rolling_max(high_ts, window=fastk_period) - tsf.rolling_min(low_ts, window=fastk_period)
    RSV = (close_ts - tsf.rolling_min(low_ts, fastk_period)) / torch.clamp(range_ts, min=1)
    K = tsf.ema(RSV, window=slowk_period)
    D = tsf.ema(K, slowd_period)
    J = 3*K - 2*D
    return RSV, K, D, J

def candleup(open_ts, close_ts, high_ts): 
    output = torch.max(open_ts, close_ts)
    output_ts = high_ts - output
    return output_ts


def candledown(open_ts, close_ts, low_ts):
    output = torch.min(open_ts, close_ts)
    output_ts = output - low_ts
    return output_ts


def atr(high_ts, low_ts, close_ts, timeperiod=14):
    true_range_ = torch.max(high_ts - low_ts, torch.abs(high_ts - tsf.shift(close_ts, window=1)))
    true_range = torch.max(true_range_, torch.abs(low_ts - tsf.shift(close_ts, window=1)))
    atr = tsf.rolling_mean_(true_range, window=timeperiod)
    return atr


def natr(high_ts, low_ts, close_ts, timeperiod=14):
    true_range_ = torch.max(high_ts - low_ts, torch.abs(high_ts - tsf.shift(close_ts, window=1)))
    true_range = torch.max(true_range_, torch.abs(low_ts - tsf.shift(close_ts, window=1)))
    TRange_max = tsf.rolling_max(true_range, window=timeperiod)
    TRange_min = tsf.rolling_min(true_range, window=timeperiod)
    natr = tsf.rolling_mean_((true_range - TRange_min) / (TRange_max - TRange_min), window=timeperiod)
    return natr

def diff(close_ts, timeperiod=12):
    diff = close_ts - tsf.rolling_mean_(tsf.shift(close_ts, timeperiod), timeperiod)
    return diff

def diff_(close_ts, timeperiod=12, max_diff_length=12):
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    output_df = pd.DataFrame(index=list(close_df.iloc[timeperiod-1:].index), columns=['signal'])
    if timeperiod <= max_diff_length:
        for i in range(len(output_df)):
            sums = 0
            for j in range(timeperiod):
                sums = sums + close_df.iloc[i+timeperiod-1].values[0] - close_df.iloc[i+j].values[0]
            output_df.iloc[i] = sums / timeperiod
    else:
        print("window size should be less than or equal to", max_diff_length)
    output_ts = torch.tensor(np.array(output_df.values, dtype='float64'), dtype=close_ts.dtype, device=close_ts.device)
    return output_ts

def dmi(high_ts, low_ts, close_ts, timeperiod=14):
    zero = torch.zeros_like(high_ts)
    upmove = high_ts - tsf.shift(high_ts, window=1)
    downmove = tsf.shift(low_ts, window=1) - low_ts

    cond_1 = upmove > downmove
    cond_2 = upmove > 0
    cond_3 = cond_1 & cond_2
    PDM = torch.where(cond_3, upmove, zero)

    cond_4 = downmove > upmove
    cond_5 = downmove > 0
    cond_6 = cond_4 & cond_5
    MDM = torch.where(cond_6, downmove, zero)

    true_range = torch.max(high_ts, tsf.shift(close_ts, window=1)) - torch.min(low_ts, tsf.shift(close_ts, window=1))
    ATR = tsf.rolling_mean_(true_range, window=timeperiod)
    PDI = 100 * tsf.ema(PDM/ATR, window=timeperiod)
    MDI = 100 * tsf.ema(MDM/ATR, window=timeperiod)
    ADX = 100 * tsf.ema(torch.abs((PDI - MDI)/(PDI + MDI)), window=timeperiod)
    ADXR = tsf.ema(ADX, window=timeperiod)
    return PDM, MDM, PDI, MDI, ADX, ADXR


def apo(close_ts, fastperiod=12, slowperiod=26, matype=0):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    apo = close_df.apply(lambda x: talib.APO(x, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype))
    
    apo_ts = torch.tensor(apo.values, dtype=close_ts.dtype, device=close_ts.device)
    return apo_ts

def cci(high_ts, low_ts, close_ts, timeperiod=20):
    TP = (high_ts + low_ts + close_ts) / 3
    mean_deviation = tsf.rolling_sum_(torch.abs(close_ts - tsf.rolling_mean_(close_ts, window=timeperiod)), window=timeperiod) / timeperiod
    cci = (TP - tsf.rolling_mean_(TP, window=timeperiod)) / (0.015 * mean_deviation)
    return cci

def cmo(close_ts, timeperiod=14):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    cmo = close_df.apply(lambda x: talib.CMO(x, timeperiod=14))
    
    CMO_ts = torch.tensor(cmo.values, dtype=close_ts.dtype, device=close_ts.device)
    return CMO_ts

def mfi(high_ts, low_ts, close_ts, volume_ts, timeperiod=14):
    TP = (high_ts + low_ts + close_ts) / 3
    MF = TP * volume_ts
    zero = torch.zeros_like(MF)
    PMF = torch.where(TP > tsf.shift(TP, window=1), MF, zero)
    NMF = torch.where(TP < tsf.shift(TP, window=1), MF, zero)
    MFR = tsf.rolling_sum_(PMF, window=timeperiod) / tsf.rolling_sum_(NMF, window=timeperiod)
    MFI = 100 - (100 / (1 + MFR))
    return MFI

def stochrsi(close_ts, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    fastk = close_df.apply(lambda x: talib.STOCHRSI(x, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[0])
    fastd = close_df.apply(lambda x: talib.STOCHRSI(x, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[1])
    fastk_ts = torch.tensor(fastk.values, dtype=close_ts.dtype, device=close_ts.device)
    fastd_ts = torch.tensor(fastd.values, dtype=close_ts.dtype, device=close_ts.device)
    return fastk_ts, fastd_ts

def trix(close_ts, timeperiod=30):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    trix = close_df.apply(lambda x: talib.TRIX(x, timeperiod=30))
    trix_ts = torch.tensor(trix.values, dtype=close_ts.dtype, device=close_ts.device)
    return trix_ts

def uos(high_ts, low_ts, close_ts, timeperiod1=7, timeperiod2=14, timeperiod3=28, timeperiod4=6):
    TH = torch.max(high_ts, tsf.shift(close_ts, window=1))
    TL = torch.min(low_ts, tsf.shift(close_ts, window=1))
    ACC1 = tsf.rolling_sum_(close_ts-TL, window=timeperiod1) / tsf.rolling_sum_(TH-TL, window=timeperiod1)
    ACC2 = tsf.rolling_sum_(close_ts-TL, window=timeperiod2) / tsf.rolling_sum_(TH-TL, window=timeperiod2)
    ACC3 = tsf.rolling_sum_(close_ts-TL, window=timeperiod3) / tsf.rolling_sum_(TH-TL, window=timeperiod3)
    UOS = (ACC1*timeperiod2*timeperiod3 + ACC2*timeperiod1*timeperiod3 + ACC3*timeperiod1*timeperiod2) * 100 / (timeperiod1*timeperiod2 + timeperiod1*timeperiod3 + timeperiod2*timeperiod3)
    MAUOS = tsf.ema(UOS, window=timeperiod4)
    return UOS, MAUOS

def wr(high_ts, low_ts, close_ts, timeperiod=14):
    HT = tsf.rolling_max(high_ts, window=timeperiod)
    LT = tsf.rolling_min(low_ts, window=timeperiod)
    WR = (HT - close_ts) / (HT - LT) * (-100)
    return WR

def dema(close_ts, timeperiod=30):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    dema = close_df.apply(lambda x: talib.DEMA(x, timeperiod=30))
    dema_ts = torch.tensor(dema.values, dtype=close_ts.dtype, device=close_ts.device)
    return dema_ts

def HT_trendline(close_ts):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    HT_trendline = close_df.apply(lambda x: talib.HT_TRENDLINE(x))
    HT_trendline_ts = torch.tensor(HT_trendline.values, dtype=close_ts.dtype, device=close_ts.device)
    return HT_trendline_ts

def kama(close_ts, timeperiod=30):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    kama = close_df.apply(lambda x: talib.KAMA(x, timeperiod=30))
    kama_ts = torch.tensor(kama.values, dtype=close_ts.dtype, device=close_ts.device)
    return kama_ts

def mama(close_ts, fastlimit=0.5, slowlimit=0.05):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    mama = close_df.apply(lambda x: talib.MAMA(x, fastlimit=0.5, slowlimit=0.05)[0])
    fama = close_df.apply(lambda x: talib.MAMA(x, fastlimit=0.5, slowlimit=0.05)[1])
    mama_ts = torch.tensor(mama.values, dtype=close_ts.dtype, device=close_ts.device)
    fama_ts = torch.tensor(fama.values, dtype=close_ts.dtype, device=close_ts.device)
    return mama_ts, fama_ts

def midpoint(close_ts, timeperiod=14):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    midpoint = close_df.apply(lambda x: talib.MIDPOINT(x, timeperiod=14))
    midpoint_ts = torch.tensor(midpoint.values, dtype=close_ts.dtype, device=close_ts.device)
    return midpoint_ts

def midprice(high_ts, low_ts, timeperiod=14):
    data = (high_ts + low_ts) / 2
    midprice = tsf.rolling_median(data, window=timeperiod)
    return midprice

def tema(close_ts, timeperiod=30):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    tema = close_df.apply(lambda x: talib.TEMA(x, timeperiod=30))
    tema_ts = torch.tensor(tema.values, dtype=close_ts.dtype, device=close_ts.device)
    return tema_ts

def wma(close_ts, timeperiod=30):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    wma = close_df.apply(lambda x: talib.WMA(x, timeperiod=30))
    wma_ts = torch.tensor(wma.values, dtype=close_ts.dtype, device=close_ts.device)
    return wma_ts

def dpo(close_ts, timeperiod=20):
    ma = tsf.rolling_mean_(close_ts, window=timeperiod)
    dpo = close_ts - tsf.shift(ma, int(timeperiod/2+1))
    return dpo

def ad(high_ts, low_ts, close_ts, volume_ts, timeperiod=5, fastperiod=3, slowperiod=10):
    zero = torch.zeros_like(high_ts, dtype=high_ts.dtype, device=high_ts.device)
    money_flow_multiplier = torch.where(high_ts == low_ts, zero, (2*close_ts-high_ts-low_ts)/(high_ts-low_ts))
    money_flow_volume = money_flow_multiplier * volume_ts
    AD_ = tsf.rolling_sum_(money_flow_volume, window=timeperiod)
    AD = tsf.shift(AD_, window=1) + money_flow_volume
    ADOSC = tsf.ema(AD, window=fastperiod) - tsf.ema(AD, window=slowperiod)
    return AD, ADOSC

def obv(close_ts, volume_ts):
    zero = torch.zeros_like(volume_ts, dtype=volume_ts.dtype, device=volume_ts.device)
    volume_new = torch.where(close_ts == tsf.shift(close_ts, window=1), zero, torch.where(close_ts > tsf.shift(close_ts, window=1), volume_ts, -volume_ts))
    OBV = tsf.shift(volume_ts, window=1) + volume_new
    OBV[0] = volume_ts[0]
    return OBV

def TSF(close_ts, timeperiod=14):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    TSF = close_df.apply(lambda x: talib.TSF(x, timeperiod=14))
    TSF_ts = torch.tensor(TSF.values, dtype=close_ts.dtype, device=close_ts.device)
    return TSF_ts

def HT_dcperiod(close_ts):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    HT_dcperiod = close_df.apply(lambda x: talib.HT_DCPERIOD(x))
    HT_dcperiod_ts = torch.tensor(HT_dcperiod.values, dtype=close_ts.dtype, device=close_ts.device)
    return HT_dcperiod_ts

def HT_dcphase(close_ts):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    HT_dcphase = close_df.apply(lambda x: talib.HT_DCPHASE(x))
    HT_dcphase_ts = torch.tensor(HT_dcphase.values, dtype=close_ts.dtype, device=close_ts.device)
    return HT_dcphase_ts

def HT_phasor(close_ts):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    inphase = close_df.apply(lambda x: talib.HT_PHASOR(x)[0])
    quadrature = close_df.apply(lambda x: talib.HT_PHASOR(x)[1])
    inphase_ts = torch.tensor(inphase.values, dtype=close_ts.dtype, device=close_ts.device)
    quadrature_ts = torch.tensor(quadrature.values, dtype=close_ts.dtype, device=close_ts.device)
    return inphase_ts, quadrature_ts

def HT_sine(close_ts):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    sine = close_df.apply(lambda x: talib.HT_SINE(x)[0])
    leadsine = close_df.apply(lambda x: talib.HT_SINE(x)[1])
    sine_ts = torch.tensor(sine.values, dtype=close_ts.dtype, device=close_ts.device)
    leadsine_ts = torch.tensor(leadsine.values, dtype=close_ts.dtype, device=close_ts.device)
    return sine_ts, leadsine_ts

def HT_trendmode(close_ts):
    import talib
    close_np = close_ts.cpu().detach().numpy()
    close_df = pd.DataFrame(close_np)
    HT_trendmode = close_df.apply(lambda x: talib.HT_TRENDMODE(x))
    HT_trendmode_ts = torch.tensor(HT_trendmode.values, dtype=close_ts.dtype, device=close_ts.device)
    return HT_trendmode_ts