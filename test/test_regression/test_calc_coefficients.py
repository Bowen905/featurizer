#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(5555)
from mock_data import create_regression_data

company_number = 2
sequence_window=20
feature_num =3
x_np, y_np, x_ts, y_ts = create_regression_data(size=company_number, sequence_window=sequence_window, feature_num=feature_num)


print("the most important thing here is understanding the shape of the dataset")
print("x shape:{}".format(x_ts.shape))
print("y shape:{}".format(y_ts.shape))
print("what the meaning of that shape?, understanding it first")

from featurizer.functions.calc_residual import get_algebra_coef_ts, get_residual_ts, calc_residual3d_ts

output_param = get_algebra_coef_ts(x=x_ts, y=y_ts)

residual_ts = get_residual_ts(x=x_ts, y=y_ts, param=output_param)

rolling_fwd_predicted_residual = calc_residual3d_ts(x_tensor=x_ts, y_tensor=y_ts, window_train=10, window_test=5, keep_first_train_nan=True, split_end=True)