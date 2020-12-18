#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np


def create_regression_data(size=2, sequence_window=30, feature_num=3):
    x_np = np.random.randn(size, sequence_window, feature_num)
    y_np = np.random.randn(size, sequence_window,1)
    x_ts = torch.tensor(x_np, dtype=torch.float32)
    y_ts = torch.tensor(y_np, dtype=torch.float32)
    return x_np, y_np, x_ts, y_ts

if __name__ == "__main__":
    x_np, y_np, x_ts, y_ts = create_regression_data()
