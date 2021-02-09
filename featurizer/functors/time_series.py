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
import torch
from featurizer.interface import Functor
from featurizer.functions import time_series_functions as tsf

class RollingMean(Functor):

    def __init__(self, window):
        super(RollingMean, self).__init__()
        self._window = window

    def forward(self, tensor):
        return tsf.rolling_mean_(tensor, window=self._window)

class RollingWeightedMean(Functor):
    
    def __init__(self, window, halflife):
        self._window = window
        self._halflife = halflife
    
    def forward(self, tensor):
        return tsf.rolling_weighted_mean(tensor, window=self._window, halflife=self._halflife)

class RollingStd(Functor):
    
    def __init__(self, window):
        super(RollingStd, self).__init__()
        self._window = window
    
    def forward(self, tensor):
        return tsf.rolling_std(tensor, window=self._window)

class RollingWeightedStd(Functor):
    
    def __init__(self, window, halflife):
        self._window = window
        self._halflife = halflife
    
    def forward(self, tensor):
        return tsf.rolling_weighted_std(tensor, window=self._window, halflife=self._halflife)
    
class RollingDownsideStd(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor, tensor_benchmark):
        return tsf.rolling_downside_std(tensor, tensor_benchmark=tensor_benchmark, window=self._window)

class RollingUpsideStd(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor, tensor_benchmark):
        return tsf.rolling_upside_std(tensor, tensor_benchmark=tensor_benchmark, window=self._window)


class RollingMeanScaledByStd(Functor):
    
    def __init__(self, window):
        super(RollingMeanScaledByStd, self).__init__()
        self._window = window
    
    def forward(self, tensor):
        return tsf.rolling_mean(tensor, window=self._window) / tsf.rolling_std(tensor, window=self._window)

class RollingCorr(Functor):
    
    def __init__(self, window):
        super(RollingCorr, self).__init__()
        self.window = window
    
    def forward(self, tensor_x, tensor_y):
        return tsf.rolling_corr(tensor_x, tensor_y, window=self.window)

class RollingSkew(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor):
        return tsf.rolling_skew(tensor, window=self._window)   

class RollingKurt(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor):
        return tsf.rolling_kurt(tensor, window = self._window)

class PctChange(Functor):

    def __init__(self, window=1):
        super(PctChange, self).__init__()
        self._window = window

    def forward(self, tensor):
        return tsf.pct_change(tensor, period=self._window)

class RollingCumulation(Functor):

    def __init__(self, window=5):
        self._window = window

    def forward(self, tensor):
        return tsf.rolling_cumulation(data_ts=tensor, window=self._window)

class RollingMax(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor):
        return tsf.rolling_max(tensor, window=self._window)

class RollingMin(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor: torch.tensor) -> torch.tensor:
        return tsf.rolling_min(tensor, window = self._window)
    
class RollingMaxDrawdown(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor: torch.tensor) -> torch.tensor:
        return tsf.rolling_max_drawdown(data_ts = tensor, window = self._window)

class RollingMaxDrawdownFromReturns(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor: torch.tensor) -> torch.tensor:
        return tsf.rolling_max_drawdown_from_returns(data_ts = tensor, window = self._window)

class RollingQuantile(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor, qscore):
        return tsf.rolling_quantile(tensor, window=self._window, qscore=qscore)
    
class RollingIdxMax(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor):
        return tsf.rolling_idxmax(tensor, widnow=self._window)
    
class RollingIdxMin(Functor):
    
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor):
        return tsf.rolling_idxmin(tensor, window=self._window)


if __name__ == "__main__":
    import torch
    
    input_data = torch.randn(20, 10)
    input_close = abs(input_data)
    
    rolling_mean_func = RollingMean(window=5)
    output1 = rolling_mean_func(input_data)