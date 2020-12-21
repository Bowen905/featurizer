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

from featurizer.interface import Functor
from featurizer.functions import time_series_functions as tsf
from featurizer.functions.calc_residual import forecast_residual3d, calc_residual3d

class ReturnsRollingStd(Functor):
    def __init__(self, window):
        self._window = window
    
    def forward(self, tensor):
        """input tensor is returns"""
        return tsf.rolling_std(tensor, window=self._window).squeeze(-1)


class Beta(Functor):
    def __init__(self, window=20):
        self._window = window

    def forward(self, tensor_x, tensor_y):
        if tensor_x.shape[-1] == 1:
            tensor_x = tensor_x.expand_as(tensor_y)
        output = tsf.rolling_cov(tensor_x, tensor_y, window=self._window) / tsf.rolling_var(tensor_x, window=self._window)
        return output


class Momentum(Functor):

    def __init__(self, window=5, lag_window=0):
        super(Momentum, self).__init__()
        self._window = window
        self._lag_window = lag_window

    def forward(self, tensor):
        """input is close"""
        tensor = tsf.pct_change(tensor, period=self._window)
        return tsf.shift(tensor, window=self._lag_window)


class BackwardSharpRatio(Functor):
    
    def __init__(self, window=30, lag_window=0):
        self._window = window
        self._lag_window = lag_window
    
    def forward(self, tensor):
        """defalt input is close"""
        returns = tsf.pct_change(tensor, period=1)
        sharp_ratio = tsf.rolling_mean_(returns, window=self._window) / tsf.rolling_std(returns, window=self._window)
        return tsf.shift(sharp_ratio, window=self._lag_window).squeeze(-1)


class AmihudRatio(Functor):
    def __init__(self, window =5):
        self._window = window
    
    def forward(self, tensor_x, tensor_y):
        """
        tensor_x is returns: sequence_window x order_book_ids
        tensor_y is turnover ratio: sequency_window x order_book_ids
        """
        output = abs(tensor_x) / tensor_y
        #pdb.set_trace()
        return tsf.rolling_mean_(output, window=self._window)


class Residual(Functor):
    
    def __init__(self, window=10):
        self._window = window
    
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return residual


class ForecastResidual(Functor):
    
    def __init__(self, window_train=10, window_test=3):
        self._window_train = window_train
        self._window_test = window_test
    
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return residual    


class ResidualRollingMean(Functor):
    """Idiosyncratic (returns) mean"""

    def __init__(self, window_train=10, window=3):
        self._window_train = window_train
        self.window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self.window_train, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_mean(residual, self._window)
    
    
class ForecastResidualRollingMean(Functor):
    """Idiosyncratic (returns) mean"""

    def __init__(self, window_train=10, window_test=3, window=3):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_mean(residual, self._window)
    
    
class ResidualRollingWeightedMean(Functor):
    """Idiosyncratic (returns) weighted mean"""

    def __init__(self, window_train=10, window=3, halflife= 90):
        self._window_train = window_train
        self._window = window
        self._halflife = halflife
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window_train, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_weighted_mean(residual, window= self._window, halflife= self._halflife)
    
    
class ForecastResidualRollingWeightedMean(Functor):
    """Idiosyncratic (returns) weighted mean"""

    def __init__(self, window_train=10, window_test=3, window=3, halflife= 90):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
        self._halflife = halflife
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_weighted_mean(residual, window= self._window, halflife= self._halflife)


class ResidualRollingStd(Functor):
    """Idiosyncratic (returns) STD"""

    def __init__(self, window_train=10, window=3):
        self._window_train = window_train
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window_train, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_std(residual, self._window)
    
    
class ForecastResidualRollingStd(Functor):
    """Idiosyncratic (returns) STD"""

    def __init__(self, window_train=10, window_test=3, window=3):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_std(residual, self._window)
    
    
class ResidualRollingWeightedStd(Functor):
    """Idiosyncratic (returns) weighted STD"""

    def __init__(self, window_train=10, window=3, halflife= 90):
        self._window_train = window_train
        self._window = window
        self._halflife = halflife
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window_train, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_weighted_std(residual, window= self._window, halflife= self._halflife)
    
    
class ForecastResidualRollingWeightedStd(Functor):
    """Idiosyncratic (returns) weighted STD"""

    def __init__(self, window_train=10, window_test=3, window=3, halflife= 90):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
        self._halflife = halflife
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_weighted_std(residual, window= self._window, halflife= self._halflife)
    
    
class ResidualRollingDownsideStd(Functor):
    """Idiosyncratic (returns) downside STD"""

    def __init__(self, window_train=10, window=3):
        self._window_train = window_train
        self._window = window
        
    def forward(self, tensor_x, tensor_y, benchmark=None):
        import torch
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window_train, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        if benchmark is None:
            benchmark = torch.zeros_like(residual)
        return tsf.rolling_downside_std(tensor=residual, tensor_benchmark= benchmark, window= self._window)
    
    
class ForecastResidualRollingDownsideStd(Functor):
    """Idiosyncratic (returns) downside STD"""

    def __init__(self, window_train=10, window_test=3, window=3):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
        
    def forward(self, tensor_x, tensor_y, benchmark=None):
        import torch
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        if benchmark is None:
            benchmark = torch.zeros_like(residual)
        return tsf.rolling_downside_std(tensor=residual, tensor_benchmark= benchmark, window= self._window)
    

class ResidualRollingUpsideStd(Functor):
    """Idiosyncratic (returns) upside STD"""

    def __init__(self, window_train=10, window=3):
        self._window_train = window_train
        self._window = window
        
    def forward(self, tensor_x, tensor_y, benchmark=None):
        import torch
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window_train, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        if benchmark is None:
            benchmark = torch.zeros_like(residual)
        return tsf.rolling_upside_std(tensor=residual, tensor_benchmark= benchmark, window= self._window)
    
    
class ForecastResidualRollingUpsideStd(Functor):
    """Idiosyncratic (returns) upside STD"""

    def __init__(self, window_train=10, window_test=3, window=3):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
        
    def forward(self, tensor_x, tensor_y, benchmark=None):
        import torch
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        if benchmark is None:
            benchmark = torch.zeros_like(residual)
        return tsf.rolling_upside_std(tensor=residual, tensor_benchmark= benchmark, window= self._window)
    

class ResidualRollingMax(Functor):
    """Idiosyncratic (returns) max"""

    def __init__(self, window_train=10, window=3):
        self._window_train = window_train
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window_train, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_max(residual, self._window)
    
    
class ForecastResidualRollingMax(Functor):
    """Idiosyncratic (returns) max"""

    def __init__(self, window_train=10, window_test=3, window=3):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_max(residual, self._window)
    
    
class ResidualRollingMin(Functor):
    """Idiosyncratic (returns) min"""

    def __init__(self, window_train=10, window=3):
        self._window_train = window_train
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window_train, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_min(residual, self._window)
    
    
class ForecastResidualRollingMin(Functor):
    """Idiosyncratic (returns) min"""

    def __init__(self, window_train=10, window_test=3, window=3):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_min(residual, self._window)
    
    
    
class ResidualRollingMedian(Functor):
    """Idiosyncratic (returns) median"""

    def __init__(self, window_train=10, window=3):
        self._window_train = window_train
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window_train, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_median(residual, self._window)
    
    
class ForecastResidualRollingMedian(Functor):
    """Idiosyncratic (returns) median"""

    def __init__(self, window_train=10, window_test=3, window=3):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_median(residual, self._window)
    

class ResidualRollingSkew(Functor):

    def __init__(self, window_train=10, window=3):
        self._window_train = window_train
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window_train, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_skew(residual, self._window)
    
    
class ForecastResidualRollingSkew(Functor):

    def __init__(self, window_train=10, window_test=3, window=3):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_skew(residual, self._window)
    

class ResidualRollingKurt(Functor):

    def __init__(self, window_train=10, window=3):
        self._window_train = window_train
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window_train, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_kurt(residual, self._window)
    
    
class ForecastResidualRollingKurt(Functor):

    def __init__(self, window_train=10, window_test=3, window=3):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_kurt(residual, self._window)
    

class ResidualRollingCumulation(Functor):
    
    def __init__(self, window_train=10, window=3):
        self._window_train = window_train
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window_train, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_cumulation(data_ts=residual, window=self._window)
    
    
class ForecastResidualRollingCumulation(Functor):
    
    def __init__(self, window_train=10, window_test=3, window=3):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_cumulation(data_ts=residual, window=self._window)


class ResidualRollingVARMOM(Functor):
    ''' 
    input is residual returns 
    VARMOM is volitility adjusted momentum; in other words, it's a Sharp Ratio for residuals with 0 degree of freedom
    '''
    
    def __init__(self, window_train=10, window=3):
        self._window_train = window_train
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window_train, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        varmom = tsf.rolling_mean_(residual, window=self._window) / tsf.rolling_std_dof_0(residual, window=self._window)
        return varmom
    
    
class ForecastResidualRollingVARMOM(Functor):
    ''' 
    input is residual returns 
    VARMOM is volitility adjusted momentum; in other words, it's a Sharp Ratio for residuals with 0 degree of freedom
    '''
    
    def __init__(self, window_train=10, window_test=3, window=3):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
        
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        varmom = tsf.rolling_mean_(residual, window=self._window) / tsf.rolling_std_dof_0(residual, window=self._window)
        return varmom
    
    
class ResidualRollingMaxDrawdownFromReturns(Functor):
    ''' input is residual returns'''
    def __init__(self, window_train=10, window=3):
        self._window_train = window_train
        self._window = window
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = calc_residual3d(tensor_x, tensor_y, window=self._window_reid, keep_first_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_max_drawdown_from_returns(data_ts=residual, window=self._window)  
    
    
class ForecastResidualRollingMaxDrawdownFromReturns(Functor):
    ''' input is residual returns'''
    def __init__(self, window_train=10, window_test=3, window=3):
        self._window_train = window_train
        self._window_test = window_test
        self._window = window
    def forward(self, tensor_x, tensor_y):
        if tensor_x.dim() < tensor_y.dim():
            tensor_x = tensor_x.expand_as(tensor_y)
        residual = forecast_residual3d(tensor_x, tensor_y, window_train=self._window_train, window_test=self._window_test, keep_first_train_nan=True)
        residual = residual.squeeze(-1).transpose(0,1)
        return tsf.rolling_max_drawdown_from_returns(data_ts=residual, window=self._window)
    


    
if __name__ == "__main__":
    import torch
    order_book_ids = 20
    sequence_window = 30
    
    # =========================================== #
    # test beta
    # =========================================== #
    returns_x = torch.randn(sequence_window, order_book_ids)
    returns_benchmark = torch.randn(sequence_window, 1)
    
    beta_functor = Beta(window=5)
    beta = beta_functor(returns_benchmark,returns_x)