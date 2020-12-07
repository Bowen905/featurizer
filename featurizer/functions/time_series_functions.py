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
from scipy.stats import rankdata
import torch
import numpy as np
import pandas as pd
from featurizer.functions.algebra_statistic import weighted_average, weighted_std, downside_std, upside_std
import pdb
# https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def rolling_sum(tensor, window=1, dim=0):
    ret = torch.cumsum(tensor, dim=dim)
    ret[window:] = ret[window:] - ret[:-window]
    ret[:window-1]= float("nan")
    return ret

def rolling_sum_(tensor, window=1, dim=0, min_periods=1):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window, min_periods=min_periods).sum()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_sum3d(tensor, window=1, dim=1):
    ret = torch.cumsum(tensor, dim=dim)
    ret[:,window:] = ret[:,window:] - ret[:,:-window]
    ret[:,:window-1]= float("nan")
    return ret

def rolling_mean(tensor, window=1):
    #to-do fixme
    #ret = torch.cumsum(tensor, dim=0)
    #ret[window:] = ret[window:] - ret[:-window]
    #ret[:window-1]= float("nan")
    #output = ret/window
    return rolling_mean_(tensor=tensor, window=window)

def rolling_mean_(tensor, window=1, min_periods=1):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window, min_periods=min_periods).mean()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_weighted_mean(tensor, window=1, min_periods=1, halflife=90):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window, min_periods=min_periods).apply(lambda x: weighted_average(x,halflife=halflife))
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

# https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def shift(tensor, window=1):
    if window == 0:
        return tensor
    
    e = torch.empty_like(tensor, dtype=tensor.dtype, device=tensor.device)
    if window > 0:
        e[:window] = float("nan")
        e[window:] = tensor[:-window]
    else:
        e[window:] = float("nan")
        e[:window] = tensor[-window:]
    return e

def diff(tensor, period=1):
    shiftd_tensor = shift(tensor, window=period)
    diff = tensor - shiftd_tensor
    return diff

def pct_change(tensor, period=1):
    shiftd_tensor = shift(tensor, window=period)
    diff = tensor - shiftd_tensor
    output = diff.div(shiftd_tensor)
    return output 

#https://stackoverflow.com/questions/54564253/how-to-calculate-the-cumulative-product-of-a-rolling-window-in-pandas
def rolling_prod(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).apply(np.prod)
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_var(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).var()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor 

def rolling_std(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).std()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_std_dof_0(tensor, window): # dof: degree of freedom
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).std(ddof=0)
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_weighted_std(tensor, window, halflife=90):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).apply(lambda x: weighted_std(x, halflife=halflife))
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_downside_std(tensor, tensor_benchmark, window):
    diff_ts = tensor - tensor_benchmark
    diff_np = diff_ts.cpu().detach().numpy()
    diff_df = pd.DataFrame(diff_np)
    output_df = diff_df.rolling(window).apply(lambda x: downside_std(x))
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_upside_std(tensor, tensor_benchmark, window):
    diff_ts = tensor - tensor_benchmark
    diff_np = diff_ts.cpu().detach().numpy()
    diff_df = pd.DataFrame(diff_np)
    output_df = diff_df.rolling(window).apply(lambda x: upside_std(x))
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_skew(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).skew()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_kurt(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).kurt()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_max(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).max()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_corr(tensor_x, tensor_y, window):
    tensor_x_np = tensor_x.cpu().detach().numpy()
    tensor_y_np = tensor_y.cpu().detach().numpy()
    tensor_x_df = pd.DataFrame(tensor_x_np)
    tensor_y_df = pd.DataFrame(tensor_y_np)
    output_df = tensor_x_df.rolling(window).corr(tensor_y_df)
    output_tensor = torch.tensor(output_df.values, dtype=tensor_x.dtype, device=tensor_x.device)
    return output_tensor

def rolling_cov(tensor_x, tensor_y, window, trailing_window=0):
    tensor_x_np = tensor_x.cpu().detach().numpy()
    tensor_y_np = tensor_y.cpu().detach().numpy()
    tensor_x_df = pd.DataFrame(tensor_x_np)
    tensor_y_df = pd.DataFrame(tensor_y_np)
    output_df = tensor_x_df.rolling(window).cov(tensor_y_df)
    output_tensor = torch.tensor(output_df.values, dtype=tensor_x.dtype, device=tensor_x.device)
    return output_tensor 

def ema(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.ewm(span=window,min_periods=window).mean()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_min(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).min()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor 

def rolling_max(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).max()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_median(tensor, window, min_periods=1):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window, min_periods=min_periods).median()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_rank(np_data):
    return rankdata(np_data,method='min')[-1]

def ts_rank(tensor, window=10):
    tensor_np = tensor.cpu().detach().numpy()
    data_df=pd.DataFrame(tensor_np)
    output_df = data_df.rolling(window).apply(rolling_rank, raw=False)
    output_np= np.array(output_df)
    output_tf=torch.tensor(output_np)
    return output_tf

def rank(tensor, axis=1, pct=True):
    tensor_np = tensor.cpu().detach().numpy()
    data_df = pd.DataFrame(tensor_np)
    output_df = data_df.rank(axis=axis, pct=pct,method='min')
    output_np = np.array(output_df)
    output_tensor = torch.tensor(output_np)
    return output_tensor

def rolling_scale(data_ts, window=10, k=1):
    output_tensor = k * data_ts / rolling_sum_(torch.abs(data_ts), window=window)
    return output_tensor

def rolling_argmax(np_data):
    return pd.DataFrame(np_data).idxmax()

def ts_argmax(tensor,window=10):
    tensor_np = tensor.cpu().detach().numpy()
    data_df = pd.DataFrame(tensor_np)
    output_df = data_df.rolling(window).apply(rolling_argmax)
    output_np = np.array(output_df)
    output_tensor = torch.tensor(output_np).squeeze()
    return output_tensor

def rolling_argmin(np_data):
    return pd.DataFrame(np_data).idxmin()

def ts_argmin(tensor,window=10):
    tensor_np = tensor.cpu().detach().numpy()
    data_df = pd.DataFrame(tensor_np)
    output_df = data_df.rolling(window).apply(rolling_argmin)
    output_np = np.array(output_df)
    output_tensor = torch.tensor(output_np).squeeze()
    return output_tensor


# ================================================================== #
# functions in financial domain                                      #
# ================================================================== #
def rolling_cumulation(data_ts: torch.tensor, window:int) -> torch.tensor:
    """the input is a variation(ratio) of something, like returns"""
    tensor_np = data_ts.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = (1+tensor_df).rolling(window).apply(np.prod, raw=False) - 1
    output_np = np.array(output_df)
    output_ts = torch.tensor(output_np).squeeze()
    return output_ts

def _max_drawdown(tensor_np: np.ndarray) -> float:
    ''' input is price '''
    expanding_max = np.maximum.accumulate(tensor_np)
    dd = (tensor_np - expanding_max) / expanding_max
    max_dd = dd.min()
    return max_dd

def rolling_max_drawdown(data_ts: torch.tensor, window: int) -> torch.tensor:
    ''' input is price '''
    tensor_np = data_ts.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).apply(_max_drawdown, raw=False)
    output_np = np.array(output_df)
    output_ts = torch.tensor(output_np).squeeze()
    return output_ts

def rolling_max_drawdown_from_returns(data_ts: torch.tensor, window: int) -> torch.tensor:
    ''' input is returns '''
    returns_np = data_ts.cpu().detach().numpy() 
    returns_df = pd.DataFrame(returns_np) 
    price = (1 + returns_df).cumprod() 
    output_df = price.rolling(window).apply(_max_drawdown, raw=False)
    output_np = np.array(output_df) 
    output_ts = torch.tensor(output_np).squeeze() 
    return output_ts
