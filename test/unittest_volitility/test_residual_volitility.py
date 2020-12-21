#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import featurizer.functors.journalhub as jh

company_num = 2
trading_dates_number = 20
feature_number = 3

y = torch.randn((company_num, trading_dates_number,1))
x = torch.randn((company_num, trading_dates_number, feature_number))

residual_func = jh.Residual(window_train=10, window_test=3)
residual = residual_func(tensor_x = x, tensor_y=y)


  
