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
from featurizer.interface import Functor
import featurizer.functions.time_series_functions as tsf
import featurizer.functions.talib_functions as talib_func

class ROCP(Functor):
    def __init__(self, timeperiod=1):
        self.timeperiod = timeperiod
    
    def forward(self, tensor):
        return tsf.pct_change(tensor)

class VolumeROCP(Functor):
    def __init__(self):
        pass
    def forward(self, tensor):
        tensor = torch.clamp(tensor, min=1)
        pct_change = tsf.pct_change(tensor)
        return torch.atan(pct_change)

class MAROCP(Functor):
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod
    
    def forward(self, tensor):
        tensor_rolling_mean = tsf.rolling_mean_(tensor, window=self.timeperiod).squeeze(-1)
        return tsf.pct_change(tensor_rolling_mean)

class MARelative(Functor):
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod
    
    def forward(self, tensor):
        tensor_rolling_mean = tsf.rolling_mean_(tensor, window=self.timeperiod).squeeze(-1)
        #tensor_rolling_mean = tensor_rolling_mean.squeeze(-1)
        relative =  (tensor_rolling_mean - tensor).div(tensor)
        return relative

class VolumeMAROCP(Functor):
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod
    
    def forward(self, tensor):
        tensor_rolling_mean = tsf.rolling_mean_(tensor, window=self.timeperiod).squeeze(-1)
        volume_pct_change =  tsf.pct_change(tensor_rolling_mean)
        return torch.atan(volume_pct_change)
    
class VolumeRelative(Functor):
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod
        
    def forward(self, tensor):
        tensor_rolling_mean = tsf.rolling_mean_(tensor, window=self.timeperiod)
        tensor_rolling_mean = tensor_rolling_mean.squeeze(-1)
        tensor = torch.clamp(tensor, min=1)
        relative =  (tensor_rolling_mean - tensor).div(tensor)
        return torch.atan(relative)

class MACDRelated(Functor):
    
    def __init__(self, fastperiod=12, slowperiod=26, signalperiod=9):
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.signalperiod = signalperiod
    
    def forward(self, tensor):
        DIF, DEA, MACD = talib_func.macd(tensor, fastperiod=self.fastperiod, slowperiod=self.slowperiod, signalperiod=self.signalperiod)
        #norm_macd = np.minimum(np.maximum(np.nan_to_num(macd), -1), 1)
        
        # norm
        norm_DIF = torch.clamp(DIF, min=-1, max=1).squeeze(-1)
        norm_DEA = torch.clamp(DEA, min=-1, max=1).squeeze(-1)
        norm_MACD = torch.clamp(MACD, min=-1, max=1).squeeze(-1)
        #pdb.set_trace()
        # diff
        norm_DIF_diff = torch.clamp(tsf.diff(DIF),min=-1, max=1).squeeze(-1)
        norm_DEA_diff = torch.clamp(tsf.diff(DEA),min=-1, max=1).squeeze(-1)
        norm_MACD_diff = torch.clamp(tsf.diff(MACD),min=-1, max=1).squeeze(-1)
        return norm_DIF, norm_DEA, norm_MACD, norm_DIF_diff, norm_DEA_diff, norm_MACD_diff

class MACDEXTRelated(Functor):
    
    def __init__(self, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0):
        self.fastperiod = fastperiod
        self.fastmatype = fastmatype
        self.slowperiod = slowperiod
        self.slowmatype = slowmatype
        self.signalperiod = signalperiod
        self.signalmatype = signalmatype
    
    def forward(self, tensor_close):
        DIF, DEA, MACD = talib_func.macdext(tensor_close, fastperiod=self.fastperiod, fastmatype=self.fastmatype, slowperiod=self.slowperiod, slowmatype=self.slowmatype, signalperiod=self.signalperiod, signalmatype=self.signalmatype)
        
        norm_DIF_ext = torch.clamp(DIF, min=-1, max=1).squeeze(-1)
        norm_DEA_ext = torch.clamp(DEA, min=-1, max=1).squeeze(-1)
        norm_MACD_ext = torch.clamp(MACD, min=-1, max=1).squeeze(-1)
        
        norm_DIF_ext_diff = torch.clamp(tsf.diff(DIF), min=-1, max=1).squeeze(-1)
        norm_DEA_ext_diff = torch.clamp(tsf.diff(DEA), min=-1, max=1).squeeze(-1)
        norm_MACD_ext_diff = torch.clamp(tsf.diff(MACD), min=-1, max=1).squeeze(-1)
        return norm_DIF_ext, norm_DEA_ext, norm_MACD_ext, norm_DIF_ext_diff, norm_DEA_ext_diff, norm_MACD_ext_diff

class MACDFIXRelated(Functor):
    
    def __init__(self, signalperiod=9):
        self.signalperiod = signalperiod
    
    def forward(self, tensor_close):
        DIF, DEA, MACD = talib_func.macdfix(tensor_close, signalperiod=9)
        
        norm_DIF_fix = torch.clamp(DIF, min=-1, max=1).squeeze(-1)
        norm_DEA_fix = torch.clamp(DEA, min=-1, max=1).squeeze(-1)
        norm_MACD_fix = torch.clamp(MACD, min=-1, max=1).squeeze(-1)
        
        norm_DIF_fix_diff = torch.clamp(tsf.diff(DIF), min=-1, max=1).squeeze(-1)
        norm_DEA_fix_diff = torch.clamp(tsf.diff(DEA), min=-1, max=1).squeeze(-1)
        norm_MACD_fix_diff = torch.clamp(tsf.diff(MACD), min=-1, max=1).squeeze(-1)
        return norm_DIF_fix, norm_DEA_fix, norm_MACD_fix, norm_DIF_fix_diff, norm_DEA_fix_diff, norm_MACD_fix_diff

class PPO(Functor):
    
    def __init__(self, fastperiod=12, slowperiod=26, matype=0):
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.matype = matype
    
    def forward(self, tensor_close):
        PPO = talib_func.ppo(tensor_close, fastperiod=self.fastperiod, slowperiod=self.slowperiod, matype=self.matype)
        return PPO

class KDJRelated(Functor):
    
    def __init__(self, fastk_period=9, slowk_period=3, slowd_period=3):
        self.fastk_period = fastk_period
        self.slowk_period = slowk_period
        self.slowd_period = slowd_period
    
    def forward(self, high, low, close):
        rsv,k,d,j = talib_func.kdj(high_ts=high,low_ts=low,close_ts=close, fastk_period=self.fastk_period, slowk_period=self.slowk_period, slowd_period=self.slowd_period)
        return rsv,k,d,j 

class RSI(Functor):
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod
    
    def forward(self, tensor):
        rsi_ts = talib_func.rsi(tensor, timeperiod=self.timeperiod).squeeze(-1)
        return rsi_ts


class DemeanedRSI(Functor):
    
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod
    
    def forward(self, tensor):
        rsi_ts = talib_func.rsi(tensor, timeperiod=self.timeperiod).squeeze(-1)
        return rsi_ts/100 -0.5 

class RSIROCP(Functor):
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod
    
    def forward(self, tensor):
        rsi_ts = talib_func.rsi(tensor, timeperiod=self.timeperiod).squeeze(-1)
        rsi_pct_change = tsf.pct_change(rsi_ts+100)
        return rsi_pct_change

class STOCHRSI(Functor):
    
    def __init__(self, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
        self.timeperiod = timeperiod
        self.fastk_period = fastk_period
        self.fastd_period = fastd_period
        self.fastd_matype = fastd_matype
    
    def forward(self, tensor_close):
        fastk, fastd = talib_func.stochrsi(tensor_close, timeperiod=self.timeperiod, fastk_period=self.fastk_period, fastd_period=self.fastd_period, fastd_matype=self.fastd_matype)
        return fastk, fastd
        

class BBANDS(Functor):
    def __init__(self, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
        self.timeperiod = timeperiod
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn
        self.matype = matype
    
    def forward(self, tensor):
        upperband_ts, middleband_ts, lowerband_ts = talib_func.bbands(tensor,timeperiod=self.timeperiod, nbdevup = self.nbdevup, nbdevdn=self.nbdevdn, matype=self.matype)
        #pdb.set_trace()
        upperband_relative_ts = (upperband_ts.squeeze(-1) - tensor)/tensor
        middleband_relative_ts = (middleband_ts.squeeze(-1) - tensor)/tensor
        lowerband_relative_ts = (lowerband_ts.squeeze(-1) - tensor)/tensor
        return upperband_relative_ts, middleband_relative_ts, lowerband_relative_ts  

class PriceVolume(Functor):
    def __init__(self):
        pass
    def forward(self, tensor_price, tensor_volume):
        rocp = tsf.pct_change(tensor_price)
        
        tensor_volume = torch.clamp(tensor_volume,min=1)
        v_pct_change = tsf.pct_change(tensor_volume)
        v_pct_change_atan = torch.atan(v_pct_change)

        pv = rocp * v_pct_change_atan
        return pv

class ReturnOvernight(Functor):
    def __init__(self, window=1):
        self._window = window
        
    def forward(self, open, close):
        shift_tensor = tsf.shift(close, window=1)
        diff = open - shift_tensor
        output = diff.div(shift_tensor)
        return output

class ReturnIntraday(Functor):
    def __init__(self, window=1):
        self._window = window
        
    def forward(self, open, close):
        diff = close - open
        output = diff.div(open)
        return output

class CandleUp(Functor):
    
    def __init__(self):
        pass
    
    def forward(self, tensor_open, tensor_close, tensor_high):
        output = talib_func.candleup(tensor_open, tensor_close, tensor_high)
        return output

class CandleDown(Functor):
    
    def __init__(self):
        pass
    
    def forward(self, tensor_open, tensor_close, tensor_low):
        output = talib_func.candledown(tensor_open, tensor_close, tensor_low)
        return output
    
class BIAS(Functor):
    def __init__(self, window):
        self._window = window
        
    def forward(self, tensor):
        bias_ts = tensor / tsf.rolling_mean(tensor, window=self._window) - 1
        return bias_ts

class ATR(Functor):
    
    def __init__(self, timeperiod=14):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_high, tensor_low, tensor_close):
        ATR = talib_func.atr(tensor_high, tensor_low, tensor_close, timeperiod=self.timeperiod)
        return ATR

class NATR(Functor):
    
    def __init__(self, timeperiod=14):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_high, tensor_low, tensor_close):
        NATR = talib_func.natr(tensor_high, tensor_low, tensor_close, timeperiod=self.timeperiod)
        return NATR

class DMIRelated(Functor):
    
    def __init__(self, timeperiod=14):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_high, tensor_low, tensor_close):
        PDM, MDM, PDI, MDI, ADX, ADXR = talib_func.dmi(tensor_high, tensor_low, tensor_close, timeperiod=self.timeperiod)
        return PDM, MDM, PDI, MDI, ADX, ADXR

class APO(Functor):
    
    def __init__(self, fastperiod=12, slowperiod=26, matype=0):
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.matype = matype
    
    def forward(self, tensor_close):
        APO = talib_func.apo(tensor_close, fastperiod=self.fastperiod, slowperiod=self.slowperiod, matype=self.matype)
        return APO

class CCI(Functor):
    
    def __init__(self, timeperiod=14):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_high, tensor_low, tensor_close):
        CCI = talib_func.cci(tensor_high, tensor_low, tensor_close, timeperiod=self.timeperiod)
        return CCI

class CMO(Functor):
    
    def __init__(self, timeperiod=14):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_close):
        CMO = talib_func.cmo(tensor_close, timeperiod=self.timeperiod)
        return CMO

class MFI(Functor):
    
    def __init__(self, timeperiod=14):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_high, tensor_low, tensor_close, tensor_total_turnover):
        MFI = talib_func.mfi(tensor_high, tensor_low, tensor_close, tensor_total_turnover, timeperiod=self.timeperiod)
        return MFI

class TRIX(Functor):
    
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_close):
        TRIX = talib_func.trix(tensor_close, timeperiod=self.timeperiod)
        return TRIX

class UOS(Functor):
    
    def __init__(self, timeperiod1=7, timeperiod2=14, timeperiod3=28, timeperiod4=6):
        self.timeperiod1 = timeperiod1
        self.timeperiod2 = timeperiod2
        self.timeperiod3 = timeperiod3
        self.timeperiod4 = timeperiod4
    
    def forward(self, tensor_high, tensor_low, tensor_close):
        UOS, MAUOS = talib_func.uos(tensor_high, tensor_low, tensor_close, timeperiod1=self.timeperiod1, timeperiod2=self.timeperiod2, timeperiod3=self.timeperiod3, timeperiod4=self.timeperiod4)
        return UOS, MAUOS

class WR(Functor):
    
    def __init__(self, timeperiod=14):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_high, tensor_low, tensor_close):
        WR = talib_func.wr(tensor_high, tensor_low, tensor_close, timeperiod=self.timeperiod)
        return WR

class DEMA(Functor):
    
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_close):
        DEMA = talib_func.dema(tensor_close, timeperiod=self.timeperiod)
        return DEMA

class EMA(Functor):
    
    def __init__(self, window):
        self.window = window
    
    def forward(self, tensor_close):
        EMA = tsf.ema(tensor_close, window=self.window)
        return EMA

class HT_TrendLine(Functor):
    
    def __init__(self):
        pass
    
    def forward(self, tensor_close):
        HT_TrendLine = talib_func.HT_trendline(tensor_close)
        return HT_TrendLine

class KAMA(Functor):
    
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_close):
        KAMA = talib_func.kama(tensor_close, timeperiod=self.timeperiod)
        return KAMA

class MAMA(Functor):
    
    def __init__(self, fastlimit=0, slowlimit=0):
        self.fastlimit = fastlimit
        self.slowlimit = slowlimit
    
    def forward(self, tensor_close):
        MAMA, FAMA = talib_func.mama(tensor_close, fastlimit=self.fastlimit, slowlimit=self.slowlimit)
        return MAMA, FAMA

class MIDPOINT(Functor):
    
    def __init__(self, timeperiod=14):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_close):
        MIDPOINT = talib_func.midpoint(tensor_close, timeperiod=self.timeperiod)
        return MIDPOINT

class MIDPRICE(Functor):
    
    def __init__(self, timeperiod=14):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_high, tensor_low):
        MIDPRICE = talib_func.midprice(tensor_high, tensor_low, timeperiod=self.timeperiod)
        return MIDPRICE

class TEMA(Functor):
    
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_close):
        TEMA = talib_func.tema(tensor_close, timeperiod=self.timeperiod)
        return TEMA

class WMA(Functor):
    
    def __init__(self, timeperiod=30):
        self.timeperiod = timeperiod
    
    def forward(self, tensor_close):
        WMA = talib_func.wma(tensor_close, timeperiod=self.timeperiod)
        return WMA

class ADRelated(Functor):
    
    def __init__(self, fastperiod=3, slowperiod=10):
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
    
    def forward(self, tensor_high, tensor_low, tensor_close, tensor_volume):
        AD, ADOSC = talib_func.ad(tensor_high, tensor_low, tensor_close, tensor_volume, fastperiod=self.fastperiod, slowperiod=self.slowperiod)
        return AD, ADOSC

class OBV(Functor):
    
    def __init__(self):
        pass
    
    def forward(self, tensor_close, tensor_volume):
        OBV = talib_func.obv(tensor_close, tensor_volume)
        return OBV

class HT_DCPERIOD(Functor):
    
    def __init__(self):
        pass
    
    def forward(self, tensor_close):
        HT_DCPERIOD = talib_func.HT_dcperiod(tensor_close)
        return HT_DCPERIOD

class HT_DCPHASE(Functor):
    
    def __init__(self):
        pass
    
    def forward(self, tensor_close):
        HT_DCPHASE = talib_func.HT_dcphase(tensor_close)
        return HT_DCPHASE

class HT_PHASOR(Functor):
    
    def __init__(self):
        pass
    
    def forward(self, tensor_close):
        inphase, quadrature = talib_func.HT_phasor(tensor_close)
        return inphase, quadrature
    

class HT_SINE(Functor):
    
    def __init__(self):
        pass
    
    def forward(self, tensor_close):
        sine, leadsine = talib_func.HT_sine(tensor_close)
        return sine, leadsine

class HT_TRENDMODE(Functor):
    
    def __init__(self):
        pass
    
    def forward(self, tensor_close):
        HT_TRENDMODE = talib_func.HT_trendmode(tensor_close)
        return HT_TRENDMODE

if __name__ == "__main__":
    pass
        
        
        
        