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
from featurizer.functions.calc_residual import forecast_residual3d

class AbsAccruals(Functor):
    
    def forward(self, net_profit, net_operate_cash_flow):
        accruals = net_profit - net_operate_cash_flow
        return abs(accruals)

class Accruals(Functor):
    
    def forward(self, net_profit, net_operate_cash_flow, capitalization):
        accruals = net_profit - net_operate_cash_flow
        return accruals/capitalization

class BookToMarket(Functor):
    
    def forward(self, pb_ratio):
        return 1/pb_ratio

class EP(Functor):
    '''Earnings to price'''
    def forward(self, pe_ratio):
        return 1/pe_ratio

class Leverage(Functor):
    
    def forward(self, total_liability, total_assets):
        return total_liability/total_assets

class Size(Functor):    
    '''Size 传入的可以是流通市值、总市值等各种代表size的指标'''
    def forward(self, size):
        output = torch.log(size)
        return output

# =========================================== #
#
# =========================================== #
class SizeNL(Functor):
    
    def __init__(self, window_train=20, window_test=5):
        self._window_train = window_train
        self._window_test = window_test
        
    def forward(self, size):
        if size.dim() == 2:
            size = size.unsqueeze(-1)
        log_size = torch.log(size)
        cube_log_size = torch.pow(log_size, 3)
        # input order in calc_residual is x,then y
        residual = forecast_residual3d(log_size, cube_log_size, window_train=self._window_train, window_test=self._window_test,keep_first_train_nan=True)
        return residual.squeeze(-1).transpose(0,1)

if __name__ == "__main__":
    torch.manual_seed(520)
    order_book_ids = 20
    sequence_window = 30
    
    tensor_x = torch.randn(sequence_window, order_book_ids)
    tensor_y = torch.randn(sequence_window, order_book_ids)
    tensor_z = abs(torch.randn(sequence_window, order_book_ids))
    
    # ======================================= #
    # Accruals                                #
    # ======================================= #
    accural_functor = Accruals()
    accrual = accural_functor(net_profit=tensor_x, net_operate_cash_flow=tensor_y, capitalization=tensor_z)
    
    # ======================================= #
    # BookToMarket                            #
    # ======================================= #
    book2market_functor = BookToMarket()
    book2market = book2market_functor(pb_ratio=tensor_x)
    
    # ======================================= #
    # Eearning2Price                          #
    # ======================================= #
    ep_functor = EP()
    ep = ep_functor(tensor_x)
    
    # ======================================= #
    # Size                                    #
    # ======================================= #
    size_functor = Size()
    size = size_functor(tensor_z)
    
    # ======================================== #
    # SizeNL
    # ======================================== #
    input_tensor = abs(torch.randn(order_book_ids, sequence_window,1)) * 100
    
    sizenl_functor = SizeNL(window_train=10, window_test=3)
    sizenl1 = sizenl_functor(input_tensor)
    sizenl2 = sizenl_functor(input_tensor.squeeze(-1))