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
import numpy as np
from functools import reduce
from featurizer.functions.split import split_sample3d

# ================================================================== #
# Analytic solution using Moore-Penrose pseudoinverse rather than    = 
# using simple multiplicative matrix inverse                         #
# ================================================================== #

def get_algebra_coef_np(x,y):
    one_arr = np.ones((*x.shape[:-1],1))
    X = np.concatenate((one_arr,x),axis=2)
    #former = np.linalg.inv(X.transpose(0,2,1)@X)
    #param = former @ X.transpose(0,2,1) @ y
    mpinv = np.linalg.pinv(X)
    param = mpinv.dot(y)[:,:,0,:]
    return param

def get_residual_np(x, y, param):
    one_arr = np.ones((*x.shape[:-1],1))
    X = np.concatenate((one_arr,x),axis=2)
    predicted = X @ param
    residual = y - predicted
    return residual

def get_algebra_coef_ts(x,y):

    """
    Parameters
    ----------
    x : TYPE torch.Tensor
        DESCRIPTION. The input tensor of size (*, m, n)(∗,m,n) where *∗ is zero or more batch dimensions.
                     in the context of finance, the * dimeson can be interpret company number, the m dimenson
                     is trading dates, the n dimenson is feature number.
    y : TYPE
        DESCRIPTION. the shape of y is (*, m, 1). this is regression task, so the label dimention is 1.

    Returns
    -------
    param_ts : TYPE
        DESCRIPTION. the shope of param_ts is (*, n+1, 1). since the feature number is n and we add intercepte to it, make it n+1.

    """
    nan_num= torch.isnan(x).sum() + torch.isnan(y).sum()
    inf = torch.isinf(x).any() or torch.isinf(y).any()
    if nan_num > 0 or inf:
        raise RuntimeError("Input tensors contain 'nan' or 'infinity'. Please process them.")
    one_arr_ts = torch.ones((*x.shape[:-1],1), device=x.device)
    X = torch.cat((one_arr_ts,x), dim=2)
    mpinv_ts = torch.pinverse(X)
    param_ts = mpinv_ts.matmul(y)
    return param_ts

def get_residual_ts(x, y, param):
    one_arr = torch.ones((*x.shape[:-1],1), device=x.device)
    X = torch.cat((one_arr,x), dim=2)
    predicted = X @ param
    residual = y - predicted
    return residual

########################################
#  Next three functions do NOT identify  
#  between train and test
########################################
def calc_residual3d_np(x_np, y_np, window=10, keep_first_nan=True):
    splitted_y = split_sample3d(y_np, window=window, step=1, offset=0, keep_tail=False, merge_remain=True)
    splitted_x = split_sample3d(x_np, window=window, step=1, offset=0, keep_tail=False, merge_remain=True)
    
    param_list = list(map(lambda x, y: get_algebra_coef_np(x, y), splitted_x, splitted_y))
    residual_list_first = get_residual_np(splitted_x[0], splitted_y[0], param_list[0])[:,:-1,:] # the first window-1 entries
    residual_list_rest = list(map(lambda x, y, p: get_residual_np(x,y,p)[:,[-1],:], splitted_x, splitted_y, param_list))  
    if keep_first_nan:
        residual_list_first.fill(np.nan)
    resid_np = reduce(lambda x,y:np.concatenate([x,y], axis=1), [residual_list_first] + residual_list_rest) 
    
    return resid_np

    
def calc_residual3d_ts(x_ts, y_ts, window=10, keep_first_nan=True):
    splitted_y = split_sample3d(y_ts, window=window, step=1, offset=0, keep_tail=False, merge_remain=True)
    splitted_x = split_sample3d(x_ts, window=window, step=1, offset=0, keep_tail=False, merge_remain=True)
    
    param_list = list(map(lambda x, y: get_algebra_coef_ts(x, y), splitted_x, splitted_y))
    residual_list_first = get_residual_ts(splitted_x[0], splitted_y[0], param_list[0])[:,:-1,:] # the first window-1 entries
    residual_list_rest = list(map(lambda x, y, p: get_residual_ts(x,y,p)[:,[-1],:], splitted_x, splitted_y, param_list))  
    if keep_first_nan:
         residual_list_first.fill_(float("nan"))
    resid_ts = reduce(lambda x,y:torch.cat([x,y], dim=1), [residual_list_first] + residual_list_rest)
    
    return resid_ts


def calc_residual3d(x, y, window=10, keep_first_nan=True):
    if isinstance(x, torch.Tensor):
        output = calc_residual3d_ts(x_ts= x, y_ts= y, window= window, keep_first_nan= keep_first_nan)
    else:
        output = calc_residual3d_np(x_np= x, y_np= y, window= window, keep_first_nan= keep_first_nan)
    return output

########################################
#  Next three functions identify  
#  between train and test
########################################
def forecast_residual3d_np(x_np, y_np, window_train=10, window_test=5, keep_first_train_nan=False, split_end=True):
    data_xy = np.concatenate((x_np, y_np), axis=2)
    # nan to num
    data_xy = np.nan_to_num(data_xy)
    
    train_xy = split_sample3d(data_xy, window=window_train, step=window_test, offset=0, keep_tail=False, merge_remain=False)
    test_xy = split_sample3d(data_xy, window=window_test, step=window_test, offset=window_train, keep_tail=False, merge_remain=True)
    
    if len(train_xy) > len(test_xy):
        if not split_end:
            train_xy = train_xy[:-abs(len(train_xy) - len(test_xy))]
        else:
            last_xy = test_xy[-1]
            splitted_last_xy = np.split(last_xy, [window_test], axis=1)
            test_xy = test_xy[:-1] + splitted_last_xy
    else:
        test_xy = test_xy[:-abs(len(train_xy) - len(test_xy))]
    
    train_x_list = [data[:, :,:-1] for data in train_xy]  # :-1
    train_y_list = [data[:, :, -1:] for data in train_xy]
    test_x_list = [data[:, :, :-1] for data in test_xy]
    test_y_list = [data[:, :, -1:] for data in test_xy]
    
    param_list = list(map(lambda x, y: get_algebra_coef_np(x, y), train_x_list, train_y_list))
    residual_train_list = list(map(lambda x, y, p: get_residual_np(x,y,p), train_x_list, train_y_list, param_list))
    residual_test_list = list(map(lambda x, y, p: get_residual_np(x,y,p), test_x_list, test_y_list, param_list))
    
    if keep_first_train_nan:
        residual_train_list[0].fill(np.nan)
    resid_np = reduce(lambda x,y:np.concatenate([x,y], axis=1), [residual_train_list[0]])
    resid_np = reduce(lambda x,y:np.concatenate([x,y], axis=1), residual_test_list, resid_np) 
        
    return resid_np


def forecast_residual3d_ts(x_tensor, y_tensor, window_train=10, window_test=5, keep_first_train_nan=False, split_end=True):
    # pdb.set_trace()
    data_xy = torch.cat((x_tensor, y_tensor), dim=2)
    
    train_xy = split_sample3d(data_xy, window=window_train, step=window_test, offset=0, keep_tail=False, merge_remain=False)
    test_xy = split_sample3d(data_xy, window=window_test, step=window_test, offset=window_train, keep_tail=False, merge_remain=True)
    
    if len(train_xy) > len(test_xy):
        if not split_end:
            train_xy = train_xy[:-abs(len(train_xy) - len(test_xy))]
        else:
            last_xy = test_xy[-1]
            splitted_last_xy = list(torch.split(last_xy, [window_test, last_xy.size()[1] - window_test], dim=1))
            test_xy = test_xy[:-1] + splitted_last_xy
    else:
        test_xy = test_xy[:-abs(len(train_xy) - len(test_xy))]

    train_x_list = [data[:, :,:-1] for data in train_xy]  # :-1
    train_y_list = [data[:, :, -1:] for data in train_xy]
    test_x_list = [data[:, :, :-1] for data in test_xy]
    test_y_list = [data[:, :, -1:] for data in test_xy]
    
    param_list = list(map(lambda x, y: get_algebra_coef_ts(x, y), train_x_list, train_y_list))
    #print('ts param shape:', param_list[0].shape)
    residual_train_list = list(map(lambda x, y, p: get_residual_ts(x,y,p), train_x_list, train_y_list, param_list))
    residual_test_list = list(map(lambda x, y, p: get_residual_ts(x,y,p), test_x_list, test_y_list, param_list))
    if keep_first_train_nan:
        residual_train_list[0].fill_(float("nan"))
    resid_ts = reduce(lambda x,y:torch.cat([x,y], dim=1), [residual_train_list[0]])
    resid_ts = reduce(lambda x,y:torch.cat([x,y], dim=1), residual_test_list, resid_ts) 
        
    return resid_ts


def forecast_residual3d(x_tensor, y_tensor, window_train=10, window_test=5, keep_first_train_nan=False):
    if isinstance(x_tensor, torch.Tensor):
        output = forecast_residual3d_ts(x_tensor=x_tensor, y_tensor=y_tensor, window_train=window_train, window_test=window_test, keep_first_train_nan=keep_first_train_nan)
    else:
        output = forecast_residual3d_np(x_np=x_tensor, y_np=y_tensor, window_train=window_train, window_test=window_test, keep_first_train_nan=keep_first_train_nan)
    return output


if __name__ == "__main__":
    import time
    np.random.seed(123)
    
    def create_data(size=1, sequence_window=30, feature_num=3):
        x_np = np.random.randn(size, sequence_window, feature_num)
        y_np = np.random.randn(size, sequence_window,1)
        x_ts = torch.tensor(x_np, dtype=torch.float32)
        y_ts = torch.tensor(y_np, dtype=torch.float32)
        return x_np, y_np, x_ts, y_ts
    
    def test_np_function(x_np, y_np):
        param_np = get_algebra_coef_np(x_np, y_np)
        residual_np = get_residual_np(x_np, y_np, param_np)
        #start_time = time.time()
        output_np =  calc_residual3d_np(x_np, y_np, window_train=20, window_test=5, keep_first_train_nan=True)
        return param_np, residual_np, output_np
    
    def test_ts_function(x_ts, y_ts):
        param_ts = get_algebra_coef_ts(x_ts, y_ts)
        residual_ts = get_residual_ts(x_ts, y_ts, param_ts)
        output_ts =  calc_residual3d_ts(x_ts, y_ts, window_train=20, window_test=5, keep_first_train_nan=True)
        return param_ts, residual_ts, output_ts
    
    def test_1_batch_data():
        pass
    
    def test_3600_batch_data():
        x_np, y_np, x_ts, y_ts = create_data(size=3600, sequence_window=60, feature_num=10)
        start_time1 = time.time()
        output1 = test_np_function(x_np, y_np)
        print("numpy function time: {}".format(time.time() - start_time1))
        
        start_time2 = time.time()
        output2 = test_ts_function(x_ts, y_ts)
        print("torch function time: {}".format(time.time() - start_time2))
        if torch.cuda.is_available():
            device = torch.device("cuda")
            x_ts.to(device)
            y_ts.to(device)
            
            start_time3 = time.time()
            output3 = test_ts_function(x_ts, y_ts)
            print("torch function with gpu accelerate time: {}".format(time.time() - start_time3))
            return output1, output2, output3
        return output1, output2
    
    output = test_3600_batch_data()