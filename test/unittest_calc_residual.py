#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:43:00 2020

@author: wanghuanqiu
"""


import unittest
import torch
from featurizer.functions.calc_residual import *
from scipy import linspace, polyval, polyfit, sqrt, stats, randn, optimize
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd


class TestOLSMethods(unittest.TestCase):
    
    def setUp(self):
    
        # ----------- Create test data required for all subsequent tests ----------------
        np.random.seed(555)
        
        self.n = int(20) #5e6
        x1 = np.linspace(-10,10,self.n)
        x2 = np.linspace(-5,3,self.n)
        self.x_2d = np.column_stack((x1, x2))
        
        # parameters:
        # a is coefficient and b is intercept
        a1=3.25; a2=2; b=-6.5
        y_fitted = a1*x1 + a2*x2 + b
        # add some noise
        self.y = y_fitted + randn(self.n)
        self.y_2d = np.expand_dims(self.y, axis= 1)

        # parameters calculated from built-in functions in Statsmodels.OLS
        self.A = sm.add_constant(self.x_2d)
        model = sm.OLS(self.y_2d, self.A)
        results = model.fit().params
        self.expected_params = np.expand_dims(results, axis=-1) # (3,1)
        
        # 2d tensors
        self.y_ts = torch.tensor(self.y)
        self.y_2d_ts = torch.stack((self.y_ts, self.y_ts))
        self.x_2d_ts = torch.tensor(self.x_2d)
        
        # 3d-x and 3d-y
        self.x_3d_ts = torch.stack((self.x_2d_ts, self.x_2d_ts)) # x_3d_ts.shape == (2,20,2)
        self.y_3d_ts = self.y_2d_ts.unsqueeze(-1) # y_3d_ts.shape == (20,2,1)
        print("The most important thing here is to understand the shape of the dataset")
        print("x shape:{}".format(self.x_3d_ts.shape))
        print("y shape:{}".format(self.y_3d_ts.shape))
        print("What's the meaning of those shapes? Understand it first.")

        
        self.error_decimal_threshold = 5
    
    
    def test_get_algebra_coef_ts(self):
        
        expected_b, expected_a1, expected_a2 = self.expected_params[:,0]
        
        # actual output
        output_coef = get_algebra_coef_ts(self.x_3d_ts, self.y_3d_ts)
        
        # check if the actual and expected parameters are almost equal
        self.assertAlmostEqual(output_coef[0,0,0].item(), expected_b, self.error_decimal_threshold, 
                               'Expected param b is different from the actual for 1st commpany')
        self.assertAlmostEqual(output_coef[1,0,0].item(), expected_b, self.error_decimal_threshold,
                               'Expected param b is different from the actual for 2nd commpany')
        self.assertAlmostEqual(output_coef[0,1,0].item(), expected_a1, self.error_decimal_threshold, 
                               'Expected param a1 is different from the actual for 1st commpany')
        self.assertAlmostEqual(output_coef[1,1,0].item(), expected_a1, self.error_decimal_threshold, 
                               'Expected param a1 is different from the actual for 2nd commpany')
        self.assertAlmostEqual(output_coef[0,2,0].item(), expected_a2, self.error_decimal_threshold,
                               'Expected param a2 is different from the actual for 1st commpany')
        self.assertAlmostEqual(output_coef[1,2,0].item(), expected_a2, self.error_decimal_threshold,
                               'Expected param a2 is different from the actual for 2nd commpany')

    
    def test_get_residual_ts(self):
        # get expected parameters and expected residuals in the desired format
        expected_params_3d_half = np.expand_dims(self.expected_params, axis=0)
        expected_params_3d = np.vstack((expected_params_3d_half, expected_params_3d_half))
        expected_residuals_2d = np.expand_dims(self.y, axis= 1) - self.A @ self.expected_params
        
        # actual output
        output_residuals = get_residual_ts(self.x_3d_ts, self.y_3d_ts, expected_params_3d)
        
        # check if the difference between the summation of the actual and the expected residuals is almost zero
        #  - print the difference if failed
        diff_2d = output_residuals[0,:,:] - expected_residuals_2d
        error_threshold = 0.1 * self.error_decimal_threshold
        self.assertTrue(abs((output_residuals[0,:,:] - expected_residuals_2d).sum()) < error_threshold, diff_2d)
    
    
    # helper function: recursively get expected residuals for the case of rolling
    def get_expected_rolling_resid_forecast(self, x_2dnp_with_constant, y, window_train, window_test, n, keep_first_train_nan, split_end):
        if split_end:
            if (n - window_train) % window_test: # the actual last test size is less than window_test
                num_rolling = (n - window_train) // window_test + 1
            else:
                num_rolling = (n - window_train) // window_test
        else: # in this case, the last test size doesn't affect number of rolling
            num_rolling = (n - window_train) // window_test
            
        end_test_size = n - window_train - window_test * (num_rolling-1)
        
        # Initialize the expected residuals differently based on whether to keep first train NaN or not
        if keep_first_train_nan:
            expected_resid = np.expand_dims(np.array([np.nan] * window_train), axis=0).T
        else:
            cur_params = sm.OLS(y[:window_train], x_2dnp_with_constant[:window_train, :]).fit().params.T
            expected_resid = np.expand_dims(y[:window_train] - x_2dnp_with_constant[:window_train, :] @ cur_params, axis=1)
        
        # Rercursively get expected residuals
        for i in range(num_rolling):
            if i == num_rolling - 1 and (n-window_train) % window_test: # last round of rolling, and when the size of data for test is less than window_test
                y_train = y[n - end_test_size - window_train : end_test_size*-1]
                y_test = y[end_test_size*-1 : ]
                x_train = x_2dnp_with_constant[n - end_test_size - window_train : end_test_size*-1, :]
                x_test = x_2dnp_with_constant[end_test_size*-1 : ]
            else:
                y_train = y[i*window_test : i*window_test+window_train]
                y_test = y[i*window_test+window_train : (i+1)*window_test+window_train]
                x_train = x_2dnp_with_constant[i*window_test : i*window_test+window_train, :]
                x_test = x_2dnp_with_constant[i*window_test+window_train : (i+1)*window_test+window_train, :]
            
            cur_params = sm.OLS(y_train, x_train).fit().params.T
            cur_resid = np.expand_dims(y_test - x_test @ cur_params, axis=-1)
            expected_resid = np.vstack((expected_resid, cur_resid)) # shape == (n,1) at last round

        return expected_resid
    
    
    # helper function: get expected residuals for calc_residual3d*, where we do not identify between train and test
    def get_expected_rolling_resid(self, x_2dnp_with_constant, y, window, n, keep_first_nan):

        num_rolling = n - window + 1
            
        # initialize expected_resid as the accumulator
        first_x, first_y = x_2dnp_with_constant[:window, :], y[:window]
        cur_params = sm.OLS(first_y, first_x).fit().params.T
        expected_resid = np.expand_dims(first_y - first_x @ cur_params, axis=1)
        if keep_first_nan:
            expected_resid[:-1,:].fill(np.nan)
            
        # recursively get expected residuals
        for i in range(1, num_rolling): # minus 1 in case last one is a different size
            cur_x, cur_y = x_2dnp_with_constant[i : i+window, :], y[i : i+window]
            cur_params = sm.OLS(cur_y, cur_x).fit().params.T
            cur_resid = np.expand_dims(cur_y - cur_x @ cur_params, axis=-1)[[-1],:]
            expected_resid = np.vstack((expected_resid, cur_resid)) 
        # # deal with the last one separately
        # last_size = len(y) - (num_rolling-1)*window
        # last_x, last_y = x_2dnp_with_constant[-1*last_size:, :], y[-1*last_size:]
        # cur_params = sm.OLS(last_y, last_x).fit().params.T
        # cur_resid = np.expand_dims(last_y - last_x @ cur_params, axis=1)
        # expected_resid = np.vstack((expected_resid, cur_resid)) # shape == (n,1) 
        
        return expected_resid
        
    def test_forecast_residual3d_ts_first_train_NaN(self):
        
        window_train, window_test = 5, 5
        '''
        manually make expected 2d residuals in numpy recursively
            - The first 5 entries should be NaN
            - Rolling should occur 3 times; Test window sizes are 5, 5, 5, respectively
        '''
        expected_resid = self.get_expected_rolling_resid_forecast(self.A, self.y, window_train, window_test, self.n, 
                                                                  keep_first_train_nan= True, split_end=True)
        expected_resid_3d_half = np.expand_dims(expected_resid, axis=0)
        expected_resid_3d = np.vstack((expected_resid_3d_half, expected_resid_3d_half))
        
        output_resid = forecast_residual3d_ts(self.x_3d_ts, self.y_3d_ts, window_train=window_train, 
                                              window_test=window_test, keep_first_train_nan= True, split_end=True)
        output_resid_np = np.array(output_resid)
        
        # Check if the difference between the expected and actual residuals sum up to almost 0
        #   - print the difference if failed
        err_threshold = 0.1 * self.error_decimal_threshold
        diff = (output_resid_np - expected_resid_3d).round(3)
        self.assertTrue(abs(np.nansum(diff)) < err_threshold, diff)
        
        
    def test_forecast_residual3d_ts_first_train_not_NaN(self):
        
        window_train, window_test = 8, 4

        '''
        manually make expected 2d residuals in numpy recursively
            - The first 10 entries should NOT be NaN
            - Rolling should occur 3 times; Test window sizes are 4, 4, 4, respectively
        '''
        expected_resid = self.get_expected_rolling_resid_forecast(self.A, self.y, window_train, window_test, self.n, 
                                                                  keep_first_train_nan=False, split_end=True)
        expected_resid_3d_half = np.expand_dims(expected_resid, axis=0)
        expected_resid_3d = np.vstack((expected_resid_3d_half, expected_resid_3d_half))
        
        output_resid = forecast_residual3d_ts(self.x_3d_ts, self.y_3d_ts, window_train=window_train, 
                                          window_test=window_test, keep_first_train_nan= False, split_end=True)
        output_resid_np = np.array(output_resid)
        
        # Check if the difference between the expected and actual residuals sum up to almost 0
        #   - print the difference if failed
        err_threshold = 0.00001 * self.error_decimal_threshold
        diff = (output_resid_np - expected_resid_3d).round(3)
        self.assertTrue(abs(np.sum(diff)) < err_threshold, diff)


    def test_forecast_residual3d_ts_irregular_last_test_size(self):
        
        window_train, window_test = 5, 4
        '''
        manually make expected 2d residuals in numpy recursively
           - The first 5 entries should be NaN
           - Rolling should occur 4 times; Test window sizes are 4, 4, 4, 3 respectively
        '''
        expected_resid = self.get_expected_rolling_resid_forecast(self.A, self.y, window_train, window_test, self.n, 
                                                                  keep_first_train_nan=True, split_end=True)
        expected_resid_3d_half = np.expand_dims(expected_resid, axis=0)
        expected_resid_3d = np.vstack((expected_resid_3d_half, expected_resid_3d_half))
        
        output_resid = forecast_residual3d_ts(self.x_3d_ts, self.y_3d_ts, window_train=window_train, 
                                              window_test=window_test, keep_first_train_nan= True, split_end=True)
        output_resid_np = output_resid.numpy()
        
        # Check if the difference between the expected and actual residuals sum up to almost 0
        #   - print the difference if failed
        err_threshold = 0.00001 * self.error_decimal_threshold
        diff = (output_resid_np - expected_resid_3d).round(3)
        self.assertTrue(abs(np.nansum(diff)) < err_threshold, diff)
        
        
    def test_forecast_residual3d_ts_irregular_last_test_size_nosplit(self):
        
        window_train, window_test = 5, 4
        '''
        manually make expected 2d residuals in numpy recursively
           - The first 5 entries should be NaN
           - Rolling should occur 3 times; Test window sizes are 4, 4, 7, respectively
        '''
        expected_resid = self.get_expected_rolling_resid_forecast(self.A, self.y, window_train, window_test, self.n, 
                                                                  keep_first_train_nan=True, split_end=False)
        expected_resid_3d_half = np.expand_dims(expected_resid, axis=0)
        expected_resid_3d = np.vstack((expected_resid_3d_half, expected_resid_3d_half))
        
        output_resid = forecast_residual3d_ts(self.x_3d_ts, self.y_3d_ts, window_train=window_train, 
                                              window_test=window_test, keep_first_train_nan= True, split_end=False)
        output_resid_np = output_resid.numpy()
        
        # Check if the difference between the expected and actual residuals sum up to almost 0
        #   - print the difference if failed
        err_threshold = 0.00001 * self.error_decimal_threshold
        diff = (output_resid_np - expected_resid_3d).round(3)
        self.assertTrue(abs(np.nansum(diff)) < err_threshold, diff)
        
        
    def test_forecast_residual3d_np(self):
        
        window_train, window_test = 5, 4
        x_3d_np = self.x_3d_ts.numpy()
        y_3d_np = self.y_3d_ts.numpy()
        '''
        The parameters are identical to the previous
        '''
        expected_resid = self.get_expected_rolling_resid_forecast(self.A, self.y, window_train, window_test, self.n, 
                                                                  keep_first_train_nan=True, split_end=False)
        expected_resid_3d_half = np.expand_dims(expected_resid, axis=0)
        expected_resid_3d = np.vstack((expected_resid_3d_half, expected_resid_3d_half))
        
        output_resid_np = forecast_residual3d_np(x_3d_np, y_3d_np, window_train=window_train, 
                                                 window_test=window_test, keep_first_train_nan= True, split_end=False)
        
        # Check if the difference between the expected and actual residuals sum up to almost 0
        #   - print the difference if failed
        err_threshold = 0.00001 * self.error_decimal_threshold
        diff = (output_resid_np - expected_resid_3d).round(3)
        self.assertTrue(abs(np.nansum(diff)) < err_threshold, diff)
        
        
    def test_forecast_residual3d_np_window_test_greater(self):
        
        window_train, window_test = 4, 7
        x_3d_np = self.x_3d_ts.numpy()
        y_3d_np = self.y_3d_ts.numpy()
        '''
        manually make expected 2d residuals in numpy recursively
           - The first 4 entries should be NaN
           - Rolling should occur 3 times; Test window sizes are 7, 7, 2, respectively
        '''
        expected_resid = self.get_expected_rolling_resid_forecast(self.A, self.y, window_train, window_test, self.n, 
                                                                  keep_first_train_nan=True, split_end=True)
        expected_resid_3d_half = np.expand_dims(expected_resid, axis=0)
        expected_resid_3d = np.vstack((expected_resid_3d_half, expected_resid_3d_half))
        
        output_resid_np = forecast_residual3d_np(x_3d_np, y_3d_np, window_train=window_train, 
                                                 window_test=window_test, keep_first_train_nan= True, split_end=True)
        
        # Check if the difference between the expected and actual residuals sum up to almost 0
        #   - print the difference if failed
        err_threshold = 0.00001 * self.error_decimal_threshold
        diff = (output_resid_np - expected_resid_3d).round(3)
        self.assertTrue(abs(np.nansum(diff)) < err_threshold, diff)
        
        
    def test_calc_residual3d_keep_fisrt_NaN(self):
        
        window = 9
        '''
        manually make expected 2d residuals in numpy recursively
           - Rolling should occur 12 times since 20 - 9 + 1 = 12
           - The first 8 entries should be NaN
        '''
        expected_resid = self.get_expected_rolling_resid(self.A, self.y, window, self.n, keep_first_nan=True)
        expected_resid_3d_half = np.expand_dims(expected_resid, axis=0)
        expected_resid_3d = np.vstack((expected_resid_3d_half, expected_resid_3d_half))
        
        output_resid_ts = calc_residual3d(self.x_3d_ts, self.y_3d_ts, window= window, keep_first_nan=True)
        output_resid_np = output_resid_ts.numpy()
        
        # Check if the difference between the expected and actual residuals sum up to almost 0
        #   - print the difference if failed
        err_threshold = 0.00001 * self.error_decimal_threshold
        diff = (output_resid_np - expected_resid_3d).round(3)
        self.assertTrue(abs(np.nansum(diff)) < err_threshold, diff)
        
        
    def test_calc_residual3d_keep_fisrt_NaN_False(self):
        
        window = 11
        x_3d_np = self.x_3d_ts.numpy()
        y_3d_np = self.y_3d_ts.numpy()
        '''
        manually make expected 2d residuals in numpy recursively
           - Rolling should occur 11 times since 20 - 11 + 1 = 10
           - No entries are NaN
        '''
        expected_resid = self.get_expected_rolling_resid(self.A, self.y, window, self.n, keep_first_nan=False)
        expected_resid_3d_half = np.expand_dims(expected_resid, axis=0)
        expected_resid_3d = np.vstack((expected_resid_3d_half, expected_resid_3d_half))
        
        output_resid_np = calc_residual3d(x_3d_np, y_3d_np, window= window, keep_first_nan=False)
        
        # Check if the difference between the expected and actual residuals sum up to almost 0
        #   - print the difference if failed
        err_threshold = 0.00001 * self.error_decimal_threshold
        diff = (output_resid_np - expected_resid_3d).round(3)
        self.assertTrue(abs(np.nansum(diff)) < err_threshold, diff)
     

if __name__ =='__main__':
    unittest.main()
    
    
    