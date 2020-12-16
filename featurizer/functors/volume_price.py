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
import featurizer.functions.volume_price as vpf
import featurizer.functions.time_series_functions as tsf

class VWAP(Functor):
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor_x, tensor_y):
        return vpf.vwap(tensor_x, tensor_y, window=self._window)

class VWAP_returns(Functor):
    def __init__(self, window=1):
        super(VWAP_returns, self).__init__()
        self._window = window
            
    def forward(self, tensor_x, tensor_y):
        """inputs are close and VWAP"""
        return (tensor_x - tsf.shift(tensor_y, window=self._window)) / tsf.shift(tensor_y, window=self._window)

class VolumeReturnsCorr(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, volume, returns):
        return tsf.rolling_corr(volume, returns, window=self._window).squeeze(-1)
    
class HighLowCorr(Functor):
    def __init__(self, window):
        self._window = window
    
    def forward(self, high, low):
        return tsf.rolling_corr(high, low, window=self._window).squeeze(-1)

class VolumeVwapDeviation(Functor):
    def __init__(self, window):
        self._window = window
        self.VWAP = VWAP(window=window)
    
    def forward(self, close, volume):
        vwap = self.VWAP(close, volume)
        #pdb.set_trace()
        return -1 * tsf.rolling_corr(vwap, volume, window=self._window).squeeze(-1)

class OpenJump(Functor):
    def __init__(self):
        pass
    def forward(self, open, close):
        """activation function"""
        return torch.atan(open/tsf.shift(close, window=1))

class AbnormalVolume(Functor):
    def __init__(self, window):
        self._window = window
    
    def forward(self, volume):
        return torch.atan(-1 * volume / tsf.rolling_mean_(volume,window=self._window).squeeze(-1))

class VolumeRangeDeviation(Functor):
    def __init__(self, window):
        self._window = window
    
    def forward(self,high, low, volume):
        return -1 * tsf.rolling_corr(high/low,volume, window=self._window).squeeze(-1)

if __name__ == "__main__":
    import torch
    size = 100
    open_ts  = abs(torch.randn(size,3))
    high_ts  = abs(torch.randn(size,3))
    low_ts  = abs(torch.randn(size,3))
    close_ts  = abs(torch.randn(size,3))
    volume_ts  = abs(torch.randn(size,3))
     
    functor = VolumeVwapDeviation(window=10)
    output = functor(close_ts,volume_ts)
    
    #functor = OpenJump()
    #output = functor(open_ts,close_ts)
    
    #functor = AbnormalVolume(window=5)
    #output = functor(volume_ts)
    
    #functor = VolumeRangeDeviation(window=8)
    #output = functor(high_ts,low_ts,volume_ts)
    
    
    