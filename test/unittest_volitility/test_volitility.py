#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from featurizer.functors.time_series import RollingDownsideStd, RollingUpsideStd

# ============================================================================== #
# understand the input shape and type of downside/upside volitility operator     #
# ============================================================================== #
strading_dates_num = 10
stock_num = 3

# such as reutrns of stocks
tensor_ts = torch.randn((strading_dates_num, stock_num))

#such as returns of index
 
benchmark = torch.randn((strading_dates_num, 1))

benchmark_expand = benchmark.expand_as(tensor_ts)


rolling_downside_std_function = RollingDownsideStd(window=5)
rolling_upside_std_function = RollingUpsideStd(window=5)


downside_volitility = rolling_downside_std_function(tensor = tensor_ts, tensor_benchmark = benchmark_expand)
upside_volitility = rolling_upside_std_function(tensor = tensor_ts, tensor_benchmark = benchmark_expand)
