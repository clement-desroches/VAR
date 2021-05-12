
"""
Created on Wed May 5 09:47:30 2021

@author: CD8590
based on the work of AD0541
code for representing portfolios, 
positions are grouped by underlying and product type
"""

import datetime as dt

import numpy as np
import pandas as pd
import scipy.stats as si


def price_option(option_type : np.ndarray, ttm: np.ndarray, strike: np.ndarray, spot: np.ndarray,
               interest: np.ndarray, vol: np.ndarray, quotity:np.ndarray, size: np.ndarray, dividend:np.ndarray):
    """ Return the price of a call option 
        Inputs : 
            - option_type (array of int) : 1 if call, -1 if put
            - ttm (int) : time to maturity
            - spot (int) : Current price of the underlying
            - strike (int) : strike price of the option
            - interest (array) : interest rate of the contract
            - vol (array) : volatility of the underlying
            - dividend (float) : dividend of the underlying
        Outputs :
            - price of the call option (float)"""
    d1 = (np.log(spot / strike) + (interest - dividend + 0.5 * vol ** 2)
          * ttm) / (vol * np.sqrt(ttm)) * option_type
    d2 = d1 - (vol * np.sqrt(ttm))*option_type
    
    option_vals = (spot * np.exp(-dividend * ttm) * si.norm.cdf(d1, 0.0, 1.0) - strike *
                   np.exp(-interest * ttm) * si.norm.cdf(d2, 0.0, 1.0))
    return option_vals * size * quotity * option_type

def price_equity(price, quantity):
    """
    Return the price of the equity position
    Inputs : 
        - price (float) : price of one equity
        - quantity (int) : quantitiy of equity in the portfolio
    Outputs :
        - price of the position (float) """
    return price*quantity

