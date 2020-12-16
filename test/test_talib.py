#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:39:08 2020

@author: hebowen
"""

import featurizer.functions.time_series_functions as tsf
import featurizer.functors.talib as talib
import featurizer.functions.talib_functions as talib_func
import torch
import numpy as np
import pandas as pd


# fake data
# type=tensor, shape=50x10
open = torch.tensor(np.random.uniform(low=20, high=40, size=(50,10)))
close = torch.tensor(np.random.uniform(low=20, high=40, size=(50,10)))
high = torch.tensor(np.random.uniform(low=30, high=50, size=(50,10)))
low = torch.tensor(np.random.uniform(low=10, high=30, size=(50,10)))
volume = torch.tensor(np.random.uniform(low=10000, high=50000, size=(50,10)))
total_turnover = torch.tensor(np.random.uniform(low=200000, high=2000000, size=(50,10)))
returns = tsf.pct_change(close, 1)

# ROCP
func_ROCP = talib.ROCP(timeperiod=1)
feature_ROCP = func_ROCP(close)

# VolumeROCP
func_VROCP = talib.VolumeROCP()
feature_VROCP = func_ROCP(volume)

# MAROCP
func_MAROCP = talib.MAROCP(timeperiod=5)
feature_MAROCP = func_MAROCP(close)

# MARelative
func_MA = talib.MARelative(timeperiod=5)
feature_MA = func_MA(close)

# VolumeMAROCP
func_VMAROCP = talib.VolumeMAROCP(timeperiod=5)
feature_VMAROCP = func_VMAROCP(volume)

# VolumeRelative
func_Volume = talib.VolumeRelative(timeperiod=5)
feature_Volume = func_Volume(volume)

# MACD
func_MACD = talib.MACDRelated(fastperiod=12, slowperiod=26, signalperiod=9)
feature_MACD = func_MACD(close)
norm_DIF = feature_MACD[0]
norm_DEA = feature_MACD[1]
norm_MACD = feature_MACD[2]
norm_DIF_diff = feature_MACD[3]
norm_DEA_diff = feature_MACD[4]
norm_MACD_diff = feature_MACD[5]

# MACDEXT
func_MACDEXT = talib.MACDEXTRelated(fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
feature_MACDEXT = func_MACDEXT(close)
norm_DIF_ext = feature_MACDEXT[0]
norm_DEA_ext = feature_MACDEXT[1]
norm_MACD_ext = feature_MACDEXT[2]
norm_DIF_diff_ext = feature_MACDEXT[3]
norm_DEA_diff_ext = feature_MACDEXT[4]
norm_MACD_diff_ext = feature_MACDEXT[5]

# MACDFIX
func_MACDFIX = talib.MACDFIXRelated(signalperiod=9)
feature_MACDFIX = func_MACDFIX(close)
norm_DIF_fix = feature_MACDFIX[0]
norm_DEA_fix = feature_MACDFIX[1]
norm_MACD_fix = feature_MACDFIX[2]
norm_DIF_diff_fix = feature_MACDFIX[3]
norm_DEA_diff_fix = feature_MACDFIX[4]
norm_MACD_diff_fix = feature_MACDFIX[5]

# PPO
func_PPO = talib.PPO(fastperiod=12, slowperiod=26, matype=0)
feature_PPO = func_PPO(close)

# KDJ
func_KDJ = talib.KDJRelated(fastk_period=9, slowk_period=3, slowd_period=3)
feature_KDJ = func_KDJ(high, low, close)
rsv = feature_KDJ[0]
k = feature_KDJ[1]
d = feature_KDJ[2]
j = feature_KDJ[3]

# RSI
func_RSI = talib.RSI(timeperiod=12)
feature_RSI = func_RSI(close)

func_DRSI = talib.DemeanedRSI(timeperiod=12)
feature_DRSI = func_DRSI(close)

func_RSIROCP = talib.RSIROCP(timeperiod=12)
feature_RSIROCP = func_RSIROCP(close)

func_STOCHRSI = talib.STOCHRSI(timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
feature_STOCHRSI = func_STOCHRSI(close)

# BBANDS
func_BBANDS = talib.BBANDS(timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
feature_BBANDS = func_BBANDS(close)
upperband_relative = feature_BBANDS[0]
middleband_relative = feature_BBANDS[1]
lowerband_relative = feature_BBANDS[2]

# PriceVolume
func_PV = talib.PriceVolume()
feature_PV = func_PV(close, volume)

# PctChange
func_pcnight = talib.PctChangeNight(window=5)
feature_pcnight = func_pcnight(open, close)

func_pcintra = talib.PctChangeIntra(window=5)
feature_pcintra = func_pcintra(open, close)

# Candle
func_CandleUp = talib.CandleUp()
feature_CandleUp = func_CandleUp(open, close, high)

func_CandleDown = talib.CandleDown()
feature_CandleDown = func_CandleDown(open, close, low)

# BIAS
func_BIAS = talib.BIAS(window=5)
feature_BIAS = func_BIAS(close)

# ATR
func_ATR = talib.ATR(timeperiod=14)
feature_ATR = func_ATR(high, low, close)

# NATR
func_NATR = talib.NATR(timeperiod=14)
feature_NATR = func_NATR(high, low, close)

# DMI
func_DMI = talib.DMIRelated(timeperiod=14)
feature_DMI = func_DMI(high, low, close)
PDM = feature_DMI[0]
MDM = feature_DMI[1]
PDI = feature_DMI[2]
MDI = feature_DMI[3]
DX = feature_DMI[4]
ADX = feature_DMI[5]
ADXR = feature_DMI[6]

# APO
func_APO = talib.APO(fastperiod=12, slowperiod=26, matype=0)
feature_APO = func_APO(close)

# CCI
func_CCI = talib.CCI(timeperiod=14)
feature_CCI = func_CCI(high, low, close)

# CMO
func_CMO = talib.CMO(timeperiod=14)
feature_CMO = func_CMO(close)

# MFI
func_MFI = talib.MFI(timeperiod=14)
feature_MFI = func_MFI(high, low, close, total_turnover)

# TRIX
func_TRIX = talib.TRIX(timeperiod=30)
feature_TRIX = func_TRIX(close)

# UOS
func_UOS = talib.UOS(timeperiod1=7, timeperiod2=14, timeperiod3=28, timeperiod4=6)
feature_UOS = func_UOS(high, low, close)
UOS = feature_UOS[0]
MAUOS = feature_UOS[1]

# WR
func_WR = talib.WR(timeperiod=14)
feature_WR = func_WR(high, low, close)

# DEMA
func_DEMA = talib.DEMA(timeperiod=30)
feature_DEMA = func_DEMA(close)

# EMA
func_EMA = talib.EMA(window=5)
feature_EMA = func_EMA(close)

# HT_TrendLine
func_HTTrend = talib.HT_TrendLine()
feature_HTTrend = func_HTTrend(close)

# KAMA
func_KAMA = talib.KAMA(timeperiod=30)
feature_KAMA = func_KAMA(close)

# MAMA
func_MAMA = talib.MAMA(fastlimit=0.5, slowlimit=0.05)
feature_MAMA = func_MAMA(close)
MAMA = feature_MAMA[0]
FAMA = feature_MAMA[1]

# Mid
func_midpoint = talib.MIDPOINT(timeperiod=14)
feature_midpoint = func_midpoint(close)

func_midprice = talib.MIDPRICE(timeperiod=14)
feature_midprice = func_midprice(high, low)

# TEMA
func_TEMA = talib.TEMA(timeperiod=30)
feature_TEMA = func_TEMA(close)

# WMA
func_WMA = talib.WMA(timeperiod=30)
feature_WMA = func_WMA(close)

# AD
func_AD = talib.ADRelated(fastperiod=3, slowperiod=10)
feature_AD = func_AD(high, low, close, volume)
AD = feature_AD[0]
ADOSC = feature_AD[1]

# OBV
func_OBV = talib.OBV()
feature_OBV = func_OBV(close, volume)

# HT_DCPERIOD
func_HTDCPERIOD = talib.HT_DCPERIOD()
feature_HTDCPERIOD = func_HTDCPERIOD(close)

# HT_DCPHASE
func_HTDCPHASE = talib.HT_DCPHASE()
feature_HTDCPHASE = func_HTDCPHASE(close)

# HT_PHASOR
func_HTPHASOR = talib.HT_PHASOR()
feature_HTPHASOR = func_HTPHASOR(close)
inphase = feature_HTPHASOR[0]
quadrature = feature_HTPHASOR[1]

# HT_SINE
func_HTSINE = talib.HT_SINE()
feature_HTSINE = func_HTSINE(close)
sine = feature_HTSINE[0]
leadsine = feature_HTSINE[1]

# HT_TRENDMODE
func_HTTrendMode = talib.HT_TRENDMODE()
feature_HTTrendMode = func_HTTrendMode(close)