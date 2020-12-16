#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import unittest
import torch
from featurizer.functions.split import *
import create_expected_output_split as ceo
import numpy as np
import pandas as pd


class TestSplitMethods(unittest.TestCase):
    
    def setUp(self):
    
        # ----------- Import and initiate expected output data ----------------
        # 2d and 3d data for testing
        self.data2d_np = ceo.data2d_np
        self.data3d_np = ceo.data3d_np
        
        # parameters consistent across functions whose input are 3d data:
        self.window = ceo.window_sample
        self.step = ceo.step_sample
        self.offset = ceo.offset_sample
        
    # helper function: convert a list of np or ts into a list of lists
    def equal_lists(self, list_np1, list_np2):
        if all(np.array_equal(i,j) for i, j in zip(list_np1, list_np2)):
            return True
        else:
            return False
    
    def test_split(self): 
        
        # expected outputs
        expected_list_split_basic = ceo.expected_list_split_basic
        expected_list_split_2steps = ceo.expected_list_split_2steps
        expected_list_split_kepttail = ceo.expected_list_split_kepttail
        expected_list_split_2offset = ceo.expected_list_split_2offset
        
        # actual outputs
        output_list_split_basic = split(self.data2d_np, window=8, step=1, offset = 0, keep_tail=False)
        output_list_split_2steps = split(self.data2d_np, window=8, step=2, offset = 0, keep_tail=False)
        output_list_split_kepttail = split(self.data2d_np, window=8, step=2, offset = 0, keep_tail=True)
        output_list_split_2offset = split(self.data2d_np, window=8, step=2, offset = 2, keep_tail=False)
        
        # check if the actual and expected parameters are almost equal
        self.assertTrue(self.equal_lists(expected_list_split_basic, output_list_split_basic), "test split() failed when step = 1")
        self.assertTrue(self.equal_lists(expected_list_split_kepttail, output_list_split_kepttail), "test split() failed when kept tail")
        self.assertTrue(self.equal_lists(expected_list_split_2offset, output_list_split_2offset), "test split() failed when offset = 2")
        self.assertTrue(self.equal_lists(expected_list_split_2steps, output_list_split_2steps), "test split() failed when step = 2")
        
    def test_split_sample(self):
        
        # expected outputs
        expected_list_split_sample_FT = ceo.expected_list_split_sample_FT
        expected_list_split_sample_FF = ceo.expected_list_split_sample_FF
        expected_list_split_sample_TT = ceo.expected_list_split_sample_TT
        expected_list_split_sample_TF = ceo.expected_list_split_sample_TF
        
        # actual outputs
        output_list_split_sample_FT = split_sample(self.data2d_np, window= self.window, step= self.step, 
                                                   offset= self.offset, keep_tail=False, merge_remain=True)
        output_list_split_sample_FF = split_sample(self.data2d_np, window= self.window, step= self.step, 
                                                   offset= self.offset, keep_tail=False, merge_remain=False)
        output_list_split_sample_TT = split_sample(self.data2d_np, window= self.window, step= self.step, 
                                                   offset= self.offset, keep_tail=True, merge_remain=True)
        output_list_split_sample_TF = split_sample(self.data2d_np, window =self.window, step= self.step, 
                                                   offset= self.offset, keep_tail=True, merge_remain=False)
        
        # check if the actual and expected parameters are almost equal
        self.assertTrue(self.equal_lists(expected_list_split_sample_FT, output_list_split_sample_FT), 
                        "test split_sample() failed when 'keep_tail' is False and 'merge_remain' True.")
        self.assertTrue(self.equal_lists(expected_list_split_sample_FF, output_list_split_sample_FF), 
                        "test split_sample() failed when 'keep_tail' is False and 'merge_remain' False.")
        self.assertTrue(self.equal_lists(expected_list_split_sample_TT, output_list_split_sample_TT), 
                        "test split_sample() failed when 'keep_tail' is True and 'merge_remain' True.")
        self.assertTrue(self.equal_lists(expected_list_split_sample_TF, output_list_split_sample_TF), 
                        "test split_sample() failed when 'keep_tail' is True and 'merge_remain' False.")
        
    def test_split3d(self):  
        
        # expected outputs
        expected_list_split3d_basic = ceo.expected_list_split3d_basic
        expected_list_split3d_2steps = ceo.expected_list_split3d_2steps
        expected_list_split3d_kepttail = ceo.expected_list_split3d_kepttail
        expected_list_split3d_2offset = ceo.expected_list_split3d_2offset
        
        # actual outputs
        output_list_split3d_basic = split3d(self.data3d_np, window=8, step=1, offset = 0, keep_tail=False)
        output_list_split3d_2steps = split3d(self.data3d_np, window=8, step=2, offset = 0, keep_tail=False)
        output_list_split3d_kepttail = split3d(self.data3d_np, window=8, step=2, offset = 0, keep_tail=True)
        output_list_split3d_2offset = split3d(self.data3d_np, window=8, step=2, offset = 2, keep_tail=False)
        
        # check if the actual and expected parameters are almost equal
        self.assertTrue(self.equal_lists(expected_list_split3d_basic, output_list_split3d_basic), "test split3d() failed when step = 1")
        self.assertTrue(self.equal_lists(expected_list_split3d_kepttail, output_list_split3d_kepttail), "test split3d() failed when kept tail")
        self.assertTrue(self.equal_lists(expected_list_split3d_2offset, output_list_split3d_2offset), "test split3d() failed when offset = 2")
        self.assertTrue(self.equal_lists(expected_list_split3d_2steps, output_list_split3d_2steps), "test split3d() failed when step = 2")

    def test_split_sample3d(self):
        
        # expected outputs
        expected_list_split_sample3d_FT = ceo.expected_list_split_sample3d_FT
        expected_list_split_sample3d_FF = ceo.expected_list_split_sample3d_FF
        expected_list_split_sample3d_TT = ceo.expected_list_split_sample3d_TT
        expected_list_split_sample3d_TF = ceo.expected_list_split_sample3d_TF
        
        # actual outputs
        output_list_split_sample3d_FT = split_sample3d(self.data3d_np, window= self.window, step= self.step, 
                                                       offset= self.offset, keep_tail=False, merge_remain=True)
        output_list_split_sample3d_FF = split_sample3d(self.data3d_np, window= self.window, step= self.step, 
                                                       offset= self.offset, keep_tail=False, merge_remain=False)
        output_list_split_sample3d_TT = split_sample3d(self.data3d_np, window= self.window, step= self.step, 
                                                       offset= self.offset, keep_tail=True, merge_remain=True)
        output_list_split_sample3d_TF = split_sample3d(self.data3d_np, window =self.window, step= self.step, 
                                                       offset= self.offset, keep_tail=True, merge_remain=False)
        
        # check if the actual and expected parameters are almost equal
        self.assertTrue(self.equal_lists(expected_list_split_sample3d_FT, output_list_split_sample3d_FT), 
                        "test split_sample3d() failed when 'keep_tail' is False and 'merge_remain' True.")
        self.assertTrue(self.equal_lists(expected_list_split_sample3d_FF, output_list_split_sample3d_FF), 
                        "test split_sample3d() failed when 'keep_tail' is False and 'merge_remain' False.")
        self.assertTrue(self.equal_lists(expected_list_split_sample3d_TT, output_list_split_sample3d_TT), 
                        "test split_sample3d() failed when 'keep_tail' is True and 'merge_remain' True.")
        self.assertTrue(self.equal_lists(expected_list_split_sample3d_TF, output_list_split_sample3d_TF), 
                        "test split_sample3d() failed when 'keep_tail' is True and 'merge_remain' False.")
    
    # test all split functions when input is tensor
    def test_splits_ts(self):
        # import input ts data
        data2d_ts = ceo.data2d_ts
        data3d_ts = ceo.data3d_ts
        
        # expected outputs
        expected_list_split_ts = ceo.expected_list_split_ts
        expected_list_split_sample_ts = ceo.expected_list_split_sample_ts
        expected_list_split3d_ts = ceo.expected_list_split3d_ts
        expected_list_split_sample3d_ts = ceo.expected_list_split_sample3d_ts
        
        # actual outputs
        output_list_split_ts = split(data2d_ts, window=8, step=2, offset = 2, keep_tail=False)
        output_list_split_sample_ts = split_sample(data2d_ts, window=self.window, step=self.step, offset = self.offset, keep_tail=False, merge_remain=False)
        output_list_split3d_ts = split3d(data3d_ts, window=8, step=2, offset = 0, keep_tail=True)
        output_list_split_sample3d_ts = split_sample3d(data3d_ts, window=self.window, step=self.step, offset = self.offset, keep_tail=True, merge_remain=True)
        
        # check if the actual and expected parameters are almost equal
        self.assertTrue(self.equal_lists(expected_list_split_ts, output_list_split_ts), 
                        "split() has problems when input is a tensor.")
        self.assertTrue(self.equal_lists(expected_list_split_sample_ts, output_list_split_sample_ts), 
                        "split_sample() has problems when input is a tensor.")
        self.assertTrue(self.equal_lists(expected_list_split3d_ts, output_list_split3d_ts), 
                        "split3d() has problems when input is a tensor.")
        self.assertTrue(self.equal_lists(expected_list_split_sample3d_ts, output_list_split_sample3d_ts), 
                        "split_sample3d() has problems when input is a tensor.")
        
if __name__ =='__main__':
    unittest.main()
    
    
    
    
    
    