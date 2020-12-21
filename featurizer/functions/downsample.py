#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import pandas as pd


"""
how to process the NaN?

in pandas.DataFrame.mean, std, skew, curt
skipna, bool, default True, Exclude NA/null values when computing the result.
"""

def downsample_mean(tensor, sample_size, axis=0):
    """
    Return sample mean over requested axis.
    Normalized by N-1 by default. This can be changed using the ddof argument


    axis{index (0), columns (1)}
    
    """
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    
    grouper = tensor_df // sample_size 
    output_df = tensor_df.groupby(by=grouper).mean(skipna=True)
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def downsample_std(tensor, sample_size, axis=0):
    """
    Return sample standard deviation over requested axis.

    Normalized by N-1 by default. This can be changed using the ddof argument


    Parameters
    ----------
    tensor : TYPE
        DESCRIPTION.
    sample_size : TYPE
        DESCRIPTION.

    Returns
    -------
    output_tensor : TYPE
        DESCRIPTION.

    """
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    
    grouper = tensor_df // sample_size 
    output_df = tensor_df.groupby(by=grouper).std(skipna=True)
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def downsample_skew(tensor, sample_size, axis=0):
    """
    Return unbiased skew over requested axis.

    Normalized by N-1.

    Parameters
    ----------
    tensor : TYPE
        DESCRIPTION.
    sample_size : TYPE
        DESCRIPTION.

    Returns
    -------
    output_tensor : TYPE
        DESCRIPTION.

    """
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    
    grouper = tensor_df // sample_size 
    output_df = tensor_df.groupby(by=grouper).mean(skipna=True)
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def downsample_kurt(tensor, sample_size, axis=0):
    """
    Return unbiased kurtosis over requested axis.

    Kurtosis obtained using Fisher’s definition of kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

    Parameters
    ----------
    tensor : TYPE
        DESCRIPTION.
    sample_size : TYPE
        DESCRIPTION.
    axis : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    output_tensor : TYPE
        DESCRIPTION.

    """
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    
    grouper = tensor_df // sample_size 
    output_df = tensor_df.groupby(by=grouper).mean(skipna=True)
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor



