# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:21:23 2019

@author: Brian

"""

import pandas as pd
import numpy as np
import os
import h5py
from functools import wraps

data_dir = ''
h5_dir = '{}'.format(data_dir)
factor_dir = '{}'.format(data_dir)


def to_hdf(data, fname, key, string=False):
    h5 = h5py.File(fname, 'a')
    if key in list(h5.keys()):
        del h5[key]
    if string:
        h5.create_dataset(key,
                          data=np.array(data, dtype=h5py.special_dtype(vlen=str)))
    else:
        h5.create_dataset(key, data=data)
    h5.close()

def to_hdf_df(df, fname, key, string=False):
    to_hdf(df, fname, key, string=string)
    to_hdf(pd.to_datetime(df.index).strftime('%Y%m%d'),
           fname, 'Dates', string=True)
    to_hdf(df.columns, fname, 'Tickers', string=True)

def from_hdf(fname, key):
    h5 = h5py.File(fname, 'r')
    data = h5[key].value
    h5.close()
    return data

def from_hdf_df(fname, key):
    return pd.DataFrame(from_hdf(fname, key),
                        index=pd.to_datetime(from_hdf(fname, 'Dates')),
                        columns=from_hdf(fname, 'Tickers'))

class TechnicalIndicators:
    """
    This class is an inherited class of TA_Indicator, which uses indicator
    functions to perform technical analysis.
    """
    def __init__(self, Data, min_ratio):
        """
        Initializing the class with an large 3-dimension numpy ndarray.
        
        Parameters
        ----------
        arr : numpy ndarray, having 3 dimensions for dates, tickers, prices,
              respectively.
              
        dates : a list of dates column, which is names for first dimension of 
                arr.
        
        tickers : a list of tickers column, which is names for second dimension
                  of arr.
                  
        prices : a list of prices column, which is names for third dimension.
        
        min_ratio : float, indicating ratio of trading days should exist when
                    calculating indicators
        
        output : directory to output factors, if None(default), nothing will
                 be output.
        """
        self.Dates = Data['Dates']
        self.Tickers = Data['Tickers']
        
        if 'Volume' not in Data:
            Data['Volume'] = Data['Turnover']*Data['MV']/Data['Close']/100
        
        na_mask = ~(Data['Volume'] != 0)
        Data['Volume'][na_mask] = np.nan
        Data['Open'][na_mask] = np.nan
        Data['High'][na_mask] = np.nan
        Data['Low'][na_mask] = np.nan
        Data['Close'][na_mask] = np.nan
        
        self.Data = Data
        
        self.min_ratio = min_ratio
        
        self.mask = ~na_mask
    
    # Decorator used for skipping non-trading days
    class SkipNoTrading(object):
        
        def __init__(self, window_pos):
            self.window_pos = window_pos
        
        def __call__(self, func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                param = list(args) + list(kwargs.values())
                origin = param[0]
                window = param[self.window_pos]
                temp_mask = origin.mask.rolling(window, axis=0).mean()
                nonskip = temp_mask > origin.min_ratio
                temp_mask = origin.mask.rolling(
                    int(np.ceil(window*(1-origin.min_ratio))),axis=0).sum()
                nonskip = nonskip & (temp_mask > 0)
                nonskip.index = origin.Dates
                nonskip.columns = origin.Tickers
                res = func(*args, **kwargs)
                if isinstance(res, list):
                    for i in range(len(res)):
                        res[i][~nonskip] = np.nan
                else:
                    res[~nonskip] = np.nan
                return res
            return wrapper
    
    @SkipNoTrading(window_pos=1)
    def SMA(self, window):
        """
        Simple Moving Average
        """
        temp = self.Data['Close'].rolling(window, axis=0).mean()
        return temp

    @SkipNoTrading(window_pos=1)
    def ROC(self, window):
        """
        Rate of Change
        
        This indicator is calculated by
        
            ROC = close(today)/close(N days ago) - 1
        
        """
        temp = self.Data['Close'].pct_change(window, axis=0)
        return temp

    @SkipNoTrading(window_pos=1)
    def std(self, window):
        """
        Standard Deviation
        """
        temp = self.Data['Close'].rolling(window, axis=0).std()
        return temp

    @SkipNoTrading(window_pos=1)
    def EMA(self, window, alpha=None):
        """
        Exponential Moving Average
        
        Parameters
        ----------
        alpha : parameter for exponential weight, the exponential weighted
                mean will be calculated as:
                    
                    alpha*[(1-alpha)**0 * x0 + (1-alpha)**1 * x1 + ...]
                    
        """
        alpha = 2.0/(window + 1.0) if alpha is None else alpha
        assert alpha <= 1 and alpha >= 0, '0 <= alpha <= 1 not satisfied!'
        temp = self.Data['Close'].ewm(alpha=alpha, axis=0).mean()
        return temp

    @SkipNoTrading(window_pos=1)
    def MOM(self, window):
        """
        Momentum
        
        To eliminate effects casted by scale, it is calculate in ratio form,
        which equals to 1 + ROC.
        
        This indicator is calculated by
            
            MOM = close(today) / close(N days ago)
        
        """
        return self.ROC(window) + 1.0

    @SkipNoTrading(window_pos=1)
    def BBands(self, window):
        """
        Bollinger Bands
        
        This indicator is calculated by
        
            std = standard deviation of close in window N
            
            middle = simple moving average of close in window N
            upper = middle + 2 * std
            lower = middle - 2 * std
            
        Returns
        -------
        [upper, middle, lower] : 3 lists in a list, which are upper band, 
                                 middle band and lower band, respectively.
                                 
        Notes
        -----
            - Typically using a 20-day window to generate the indicators.
            
            - Approximately 90%(for Gaussian distribution, 95%) of price
              action occurs between the upper and lower band.
             
            - They are simply indicators indicate price volatility, breaking
              out bands doesn't mean to be a signal.
        """
        middle = self.SMA(window)
        std = self.std(window)
        upper = middle + 2*std
        lower = middle - 2*std
        
        return [upper, middle, lower]

    @SkipNoTrading(window_pos=1)
    def ATR(self, window):
        """
        Average True Range
        
        This indicator is calculated by
            
            TR[i] = max{high[i] - low[i],
                        abs(high[i] - close[i-1]),
                        abs(low[i] - close[i-1])}
            
            ATR = EMA(TR, alpha = 1 - 1/N)
        
        Notes
        -----
            - Typically using a 14-day window to generage the indicator.
            
            - This indicator show the commitment of enthusiasm of traders.
            
            - Larger range suggests to higher interest to continue bid up
              or sell down a stock.
        """
        temp1 = self.Data['High'] - self.Data['Low']
        temp2 = (self.Data['High'] - self.Data['Close'].shift(axis=0)).abs()
        t_mask = temp1 < temp2
        temp1[t_mask] = temp2
        temp2 = (self.Data['Low'] - self.Data['Close'].shift(axis=0)).abs()
        t_mask = temp1 < temp2
        temp1[t_mask] = temp2
        temp1.ewm(alpha=1-1/window, axis=0).mean()
        
        return temp1

    @SkipNoTrading(window_pos=1)
    def STO(self, window):
        """
        Stochastic Oscillator
        
        This indicator is calculated by
            
            %K = 100 * (close - N-day low) / (N-day high - N-day low)
            %D = (K1 + K2 + K3) / 3
        
        Returns
        -------
        [%K, %D] : 2 lists in a list, which are %K and %D.
        
        Notes
        -----
            - The indicators are in percentage(%) scale.
            
            - Typical window sizes for the 2 indicators are 5, 9, 14.
            
            - Traditional settings use 80 as the overbought threshold and 20
              as the oversold threshold.
        """
        nd_high = self.Data['High'].rolling(window, axis=0).max()
        nd_low = self.Data['Low'].rolling(window, axis=0).min()
        STOK = (self.Data['Close'] - nd_low) / (nd_high - nd_low)
        STOD = pd.DataFrame(STOK).rolling(3, axis=0).mean()
        return [STOK, STOD]

    @SkipNoTrading(window_pos=1)
    def TRIX(self, window):
        """
        Trix
        
        This indicator is calculated by
        
            TRIX = pct_change(EMA(EMA(EMA(close))))
        
        Notes
        -----
            - Like any moving average, the triple EMA is just a smoothing of
              price data, and therefore is trend-following.
             
            - The standard setting for it is 15 for the triple smoothed EMA
              and 9 for the signal line(EMA of TRIX)
        """
        smooth = self.Data['Close'].ewm(span=window, axis=0).mean()
        smooth = smooth.ewm(span=window, axis=0).mean()
        smooth = smooth.ewm(span=window, axis=0).mean()
        temp = smooth.pct_change(axis=0)
        return temp

    @SkipNoTrading(window_pos=1)
    def ADX(self, window, nadx):
        """
        Average Directional Movement Index
        
        This indicator is calculated by
        
            UpMove = high(today) - high(yesterday)
            DownMove = low(yesterday) - low(today)
            
            if UpMove > DownMove and Upmove > 0, then +DM = UpMove, else 0
            if DownMove > UpMove and DownMove > 0, then -DM = DownMove, else 0
            
            +DI = 100 * EMA(+DM) / ATR
            -DI = 100 * EMA(-DM) / ATR
            
            ADX = 100 * EMA(abs(+DI - -DI) / (+DI + -DI), nadx)
            
        Notes
        -----
            - This indicator does not indicate trend direction or momentum,
              only trend strength.
              
            - Generally, ADX readings below 20 indicate trend weakness, and
              readings above 40 indicate trend strength.
              
            - The standard setting for this indicator is 10 for window size
              and the same for smoothing window.
        """
        up = self.Data['High'] - self.Data['High'].shift(axis=0)
        down = self.Data['Low'].shift(axis=0) - self.Data['Low'].shift(axis=0)
        
        pos_dm = (up > down) & (up > 0)
        neg_dm = (down > up) & (down > 0)
        
        up[~pos_dm] = 0
        down[~neg_dm] = 0
        
        ATR = self.ATR(window)
        
        pos_di = 100.0*pos_dm.ewm(span=window, axis=0).mean() / ATR
        neg_di = 100.0*neg_dm.ewm(span=window, axis=0).mean() / ATR
        
        ADX = 100.0*((pos_di - neg_di).abs() /
                     (pos_di + neg_di)).ewm(span=nadx, axis=0).mean()
        
        return [ADX, pos_di, neg_di]

    @SkipNoTrading(window_pos=1)
    def MACD(self, n_short, n_long):
        """
        MACD, MACD Signal and MACD difference
        
        This indicator is calculated by
        
            DIF = EMA(close, n_short) / EMA(close, n_long)
            DEM = EMA(DIF, 9)
            OSC = DIF - DEM
        
        Returns
        -------
        [DIF, DEM, OSC] : 3 lists in a list, which is MACD line, signal line
                          and MACD histogram, respectively.
        
        Notes
        -----
            - The standard setting for MACD is 12 for short-term window, 26 
              for long-term window and 9 for smoothing window.
              
            - Positive MACD increase as the short-term EMA diverges further
              from the long-term one, which indicates upside momentum is
              increasing.
        """
        EMAshort = self.Data['Close'].ewm(span=n_short, axis=0).mean()
        EMAlong = self.Data['Close'].ewm(span=n_long, axis=0).mean()
        
        DIF = EMAshort - EMAlong
        DEM = DIF.ewm(span=9, axis=0).mean()
        OSC = DIF - DEM
        
        return [DIF, DEM, OSC]

    @SkipNoTrading(window_pos=1)
    def RSI(self, window):
        """
        Relative Strength Index
        
        This indicator is calculated by
            
            if close[i] > close[i-1]:
                U = close[i] - close[i-1]
                D = 0
            else:
                U = 0
                D = close[i-1] - close[i]
                
            RS = SMA(U, N) / SMA(D, N)
            RSI = 100 - 100/(1 + RS)
        
        Notes
        -----
            - The default lookback period for RSI is 14, but this can be
              lowered to increase sensitivity or raised to decrease
              sensitivity.
            
            - RSI is considered overbought when above 70 and oversold when
              below 30.
        """
        ret = self.Data['Close'].diff(axis=0)
        
        U = ret.abs()
        D = ret.abs()
        U[~(ret>0)] = 0
        D[~(ret<0)] = 0
        
        RS = (U.ewm(span=window, axis=0).mean()/
              D.ewm(span=window, axis=0).mean())
        RSI = (1 - 1/(1+RS))*100.0
        
        return RSI

    @SkipNoTrading(window_pos=1)
    def OBV(self, window):
        """
        On-balance Volume
        
        This indicator is calculated by
            
            if close[i] > close[i-1]:
                dV = volume
            elif close[i] == close[i-1]:
                dV = 0
            else:
                dV = -volume
            
            OBV[i] = OBV[i-1] + dV
        
        Notes
        -----
            - Volumes used in this function are scaled by free market value.
            
            - A bullish divergence forms when OBV moves higher or forms a
              higher low even as prices move lower or forge a lower low.
        """
        temp = np.sign(self.Data['Close'].diff(axis=0))
        OBV = temp * self.Data['Volume'] / self.Data['fMV'] * 10000
        OBV = OBV.rolling(window, axis=0).mean()
        return OBV

class TechnicalAnalysis(TechnicalIndicators):
    
    def __init__(self, min_ratio=0.75, output = None, smooth = False):
        Dates = from_hdf('{}/Close.h5'.format(h5_dir), 'Dates')
        Data = {
                'Dates': pd.to_datetime(Dates),
                'Tickers': from_hdf('{}/Close.h5'.format(h5_dir), 'Tickers'),
                'Open': from_hdf_df(
                        '{}/Open.h5'.format(h5_dir), 'Forward'),
                'High': from_hdf_df(
                        '{}/High.h5'.format(h5_dir), 'Forward'),
                'Low': from_hdf_df(
                        '{}/Low.h5'.format(h5_dir), 'Forward'),
                'Close': from_hdf_df(
                        '{}/Close.h5'.format(h5_dir), 'Forward'),
                'VWAP': from_hdf_df(
                        '{}/VWAP.h5'.format(h5_dir), 'Forward'),
                'MV': from_hdf_df(
                        '{}/MV.h5'.format(h5_dir), 'Raw'),
                'Turnover': from_hdf_df(
                        '{}/Turnover.h5'.format(h5_dir), 'Raw'),
                'fMV': from_hdf_df(
                        '{}/fMV.h5'.format(h5_dir), 'Raw'),
                'fTurnover': from_hdf_df(
                        '{}/fTurnover.h5'.format(h5_dir), 'Raw'),
                }
        super().__init__(Data, min_ratio)
        
        self.output = output
        self.smooth = smooth
    
    class Output(object):
        def __init__(self):
            pass
        
        def __call__(self, func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                origin = list(args)[0]
                smooth = origin.smooth
                output = origin.output
                output_name = func.__name__.split('_')[1:]
                param = list(args) + list(kwargs.values())
                output_name += [str(int(x)) for x in param[1:]]
                output_name = '_'.join(output_name)
                res = func(*args, **kwargs)
                if origin.output is not None:
                    if not os.path.exists(output):
                        os.makedirs(output)
                    to_hdf_df(res,
                              '{}/{}.h5'.format(origin.output, output_name),
                              'Raw')
                    if smooth:
                        to_hdf_df(res.ewm(alpha=0.8, axis=0).mean(),
                              '{}/{}.h5'.format(origin.output, output_name),
                              'EMA')
                print('{} Done!'.format(output_name))
                return res
            return wrapper
    
    @Output()
    def Factor_RSI_bot(self, short, rolling):
        """
        Summation of distances of short-term RSI from long-term RSI, 
        30 and minimal value among rolling period.
        
        Parameters
        ----------
        short : integer, length for short term RSI period.
                
        rolling : integer, length for minimal value period.
        """
        ind_short = self.RSI(short)        
        temp_short = ind_short.fillna(100)        
        support = temp_short.rolling(rolling, axis=0).min()
        alpha = (30-temp_short)/(1-support+temp_short)
        return alpha
    
    @Output()
    def Factor_RSI_low(self, window):
        """
        Buying stocks with RSI < 30. RSI will be exponentially smoothed.
        
        Parameters
        ----------
        window : integer, length for RSI period.
        
        Notes
        -----
            - Threshold for RSI can be set to other values.
            
            - Factor scores will be smoothed using exponential mean with span
              equal to length.
        """
        ind = self.RSI(window)
        alpha = 100-ind
        return alpha
    
    @Output()
    def Factor_RSI_comp(self, short, long, rolling):
        """
        Summation of distances of short-term RSI from long-term RSI, 
        30 and minimal value among rolling period.
        
        Parameters
        ----------
        short : integer, length for short term RSI period.
        
        long : integer, length for long term RSI period.
        
        rolling : integer, length for minimal value period.
        """
        ind_short = self.RSI(short)
        ind_long = self.RSI(long)

        support = ind_short.rolling(rolling, axis=0).min()
        alpha = ind_long+support+30 - 3*ind_short
        return alpha
    
    @Output()
    def Factor_MACD(self, ss, sl, ls, ll, rolling):
        """
        Betting on extreme reversal condition using MACD.
        
        Parameters
        ----------
        ss : integer, length for short term SMA used in short term MACD
        
        sl : integer, length for long term SMA used in short term MACD
        
        ls : integer, length for short term SMA used in long term MACD
        
        ll : integer, length for long term SMA used in long term MACD
        
        rolling : integer, length for minimal value period.
        
        Notes
        -----
            - Since all parameters are opened to be self defined, this factor
              will generate many versions.
        """
        ind_short = self.MACD(ss, sl)
        ind = self.MACD(ls, ll)
        
        temp_short = ind_short[2].fillna(0)
        temp_long = ind[2].fillna(0)
        
        support = temp_short.rolling(rolling,axis=0).min()
        alpha = support + temp_long - 2*temp_short
        return alpha
    
    @Output()
    def Factor_TRIX(self, short, long, rolling):
        """
        Betting on extreme reversal condition using TRIX.
        
        Paramters
        ---------
        short : integer, length for short term TRIX
        
        long : integer, length for long term TRIX
        
        rolling : integer, length for minimal value period
        
        Notes
        -----
            - Different from MACD reversal, this factor has a double-check on
              long-term minimal values.
        """
        ind_short = self.TRIX(short)
        ind = self.TRIX(long)
        support = ind.rolling(rolling,axis=0).min()
        support_short = ind_short.rolling(rolling,axis=0).min()
        alpha = support+support_short+ind-3*ind_short
        
        return alpha
    
    @Output()
    def Factor_TRIX_ret(self, short, long, rolling):
        """
        Betting on extreme TRIX reversal along with large drawdown.
        
        Parameters
        ----------
        short : integer, length for short term TRIX
        
        long : integer, length for long term TRIX
        
        rolling : integer, length for minimal value period
        """
        ind_short = self.TRIX(short)
        ind = self.TRIX(long)
        ret = self.ROC(short)
        support = ind.rolling(rolling,axis=0).min()
        support_short = ind_short.rolling(rolling,axis=0).min()
        support_flag = ret.rolling(rolling,axis=0).min()
        alpha = (support+support_short+ind-3*ind_short)/(1+support_flag-ret)
        
        return alpha
    
    @Output()
    def Factor_TRIX_vol(self, short, long, rolling):
        """
        Betting on extreme TRIX reversal along with large sell volume.
        
        Parameters
        ----------
        short : integer, length for short term TRIX
        
        long : integer, length for long term TRIX
        
        rolling : integer, length for minimal value period
        
        Notes
        -----
            - Volume used in this factor is OBV indicator, which is a summation
              of volume with directions.
        """
        ind_short = self.TRIX(short)
        ind = self.TRIX(long)
        vol = self.OBV(short)
        s_vol = vol.sub(vol.min(axis=1),axis=0).div(vol.max(axis=1).sub(
                vol.min(axis=1),axis=0),axis=0)
        support = ind.rolling(rolling,axis=0).min()
        support_short = ind_short.rolling(rolling,axis=0).min()
        alpha = support+support_short+ind-3*ind_short
        alpha = alpha.sub(alpha.min(axis=1),axis=0).div(alpha.max(axis=1).sub(
                alpha.min(axis=1),axis=0),axis=0)
        alpha *= np.exp(1 + s_vol)    
        
        return alpha
    
    @Output()
    def Factor_BBands(self, window):
        """
        Betting on reversal of distance between BBands lower bound and
        close prices.
        
        Parameters
        ----------
        window : integer, input parameter for BBands
                
        Notes
        -----
            - Due to the reason that sudden drawdown will 
        """
        lower = self.BBands(window)[-1]
        dist = self.Data['Close'] - lower
        dist = dist.sub(dist.min(axis=1),axis=0).div(dist.max(axis=1).sub(
                dist.min(axis=1),axis=0),axis=0)
        opf = self.Data['Close']
        opf = opf.sub(opf.min(axis=1),axis=0).div(opf.max(axis=1).sub(
                opf.min(axis=1),axis=0),axis=0)
        history = (~np.isnan(self.Data['Close'])).astype('float16').cumsum(axis=0)
        history = history.sub(history.min(axis=1),axis=0).div(
                history.max(axis=1).sub(history.min(axis=1),axis=0),axis=0)
        alpha = 1/(1 + dist*opf*history)
        
        return alpha
    
    @Output()
    def Factor_BBandsD(self, short, long):
        """
        Betting on reversal of distance between BBands lower bound and
        close prices.
        
        Parameters
        ----------
        short : integer, length for short term BBands
        
        long : integer, length for long term BBands
        
        Notes
        -----
            - Due to the reason that sudden drawdown will 
        """
        lower_short = self.BBands(short)[-1]
        lower = self.BBands(long)[-1]
        dist = (lower - self.Data['Close']).ewm(span=long, axis=0).mean()
        dist_short = (lower_short - self.Data['Close']).ewm(span=short,
                     axis=0).mean()
        alpha = dist_short - dist
        
        return alpha
    
    @Output()
    def Factor_ADX(self, window, rolling):
        """
        Betting on extreme reversal of distance between upside movement and
        downside movement, with an enhancement from trend strength.
        
        Parameters
        ---------- 
        window : integer, input parameter for ROC
        
        rolling : integer, length for minimal value period
        
        Notes
        -----
            - For the reason that ADX is an indicator for trend strength, 
              sometimes the trend strengh signal may not sensitive enough
              due to its correlation to ATR indicator.
        """
        ind = self.ADX(window, 5)
        ret = self.ROC(window)
        gap_DI = ind[1] - ind[2]
        support = gap_DI.rolling(rolling,axis=0).min()
        dist = gap_DI - support
        dist = dist.sub(dist.min(axis=1),axis=0).div(dist.max(axis=1).sub(
                dist.min(axis=1),axis=0),axis=0)
        adx = 100-ind[0]
        adx = adx.sub(adx.min(axis=1),axis=0).div(adx.max(axis=1).sub(
                adx.min(axis=1),axis=0),axis=0)
        ret = ret.sub(ret.mean(axis=1),axis=0).div(ret.std(axis=1),axis=0)
        alpha = -ret / (1+dist)
        
        return alpha
    
    @Output()
    def Factor_Reversal(self, window):
        """
        Traditional reversal factor which longs stocks with lowest returns
        last time period.
        
        Parameters
        ----------
        window : integer, input parameter for ROC
        """
        alpha1 = self.Data['Close'].pct_change(axis=0)
        alpha = (alpha1.rolling(window, axis=0).max() -
                 alpha1.rolling(window, axis=0).min())
        
        return alpha

    @Output()
    def Factor_MOM(self, window):
        """
        Traditional momentum factor which longs stocks having highest returns
        last time period.
        
        Parameters
        ----------
        window : integer, input parameter for MOM
        """
        alpha = self.MOM(window)
        
        return alpha
    
    @Output()
    def Factor_MAD(self, short, long):
        """
        Moving average distance with simple trend confirmation.
        
        Parameters
        ----------
        short : integer, length for short term SMA
        
        long : integer, length for long term SMA
        
        Notes
        -----
            - This factor will be exponentially smoothed with span = 5.
        """
        sma = self.SMA(short)
        sma_long = self.SMA(long)
        mad = sma / sma_long
        mad2 = mad.shift(axis=0)
        mad3 = mad2.shift(axis=0)
        alpha = (mad2 > mad)*(mad2 - mad)*(mad3 > mad2)
        alpha = alpha.ewm(span=5, axis=0).mean()
        return alpha
    
    @Output()
    def Factor_MAD_std(self, short, long):
        """
        Ratio of MAD for close prices and MAD for standard deviations, with
        simple trend confirmation.
        
        Parameters
        ----------
        short : integer, length for short term indicators
        
        long : integer, length for short term indicators
        
        Notes
        -----
            - This factor will be exponentially smoothed with span = 5.
        """
        ind = self.SMA(short) / self.SMA(long)
        std = self.std(short) / self.std(long)
        ind2 = ind.shift(axis=0)
        ind3 = ind2.shift(axis=0)
        flag = std / std.shift(axis=0)
        alpha = (ind2 > ind)*(ind2 - ind)*(ind3 > ind2)*flag
        alpha = alpha.ewm(span=5,axis=0).mean()
        return alpha

    @Output()
    def Factor_STO_TRIX(self, window):
        """
        Betting on large spread between from %D and %K.
        
        Parameters
        ----------
        window : integer, input parameter for STO
        
        rolling : integer, length for maximal value period
        
        Notes
        -----
            - This factor will be exponentially smoothed with span = 5.
        """
        alpha = self.STO(window)[0]
        alpha = alpha.ewm(span=window, axis=0).mean()
        alpha = alpha.ewm(span=window, axis=0).mean()
        alpha = -alpha.ewm(span=window, axis=0).mean()
        return alpha
    
    @Output()
    def Factor_STO(self, window, rolling):
        """
        Betting on large spread between from %D and %K.
        
        Parameters
        ----------
        window : integer, input parameter for STO
        
        rolling : integer, length for maximal value period
        
        Notes
        -----
            - This factor will be exponentially smoothed with span = 5.
        """
        ind = self.STO(window)
        flag = ind[1] - ind[0]
        support = flag.rolling(rolling, axis=0).max()
        alpha = flag - support
        alpha = alpha.ewm(span=5, axis=0).mean()
        return alpha
    
    @Output()
    def Factor_OBV(self, short, long, rolling):
        """
        Betting on extremely low values of directional volume.
        
        Parameters
        ----------
        short : integer, length for short term OBV
        
        long : integer, length for long term OBV
        
        rolling : integer, length for minimal value period
        
        Notes
        -----
            - This factor will be exponentially smoothed with span = 5.
        """
        ind_short = self.OBV(short)
        ind = self.OBV(long)
        support = ind_short.rolling(rolling,axis=0).min()
        alpha = support+ind-2*ind_short
        alpha = alpha.ewm(span=5,axis=0).mean()
        return alpha
    
    @Output()
    def Factor_OBV2(self, short, long, rolling):
        """
        Betting on extremely low values of directional volume when the short
        term line cross the long term line yesterday.
        
        Parameters
        ----------
        short : integer, length for short term OBV
        
        long : integer, length for long term OBV
        
        rolling : integer, length for minimal value period
        
        Notes
        -----
            - This factor will be exponentially smoothed with span = 5.
        """
        ind_short = self.OBV(short)
        ind = self.OBV(long)
        support = ind_short.rolling(rolling,axis=0).min()
        alpha = (support - ind_short)*(ind>ind_short)*(ind.shift(axis=0)>
                 ind_short.shift(axis=0))
        alpha = alpha.ewm(span=5,axis=0).mean()
        return alpha
    
    @Output()
    def Factor_CProb(self, short, long, lookback):
        """
        Use correlation coefficients to confirm recent upside trend. The
        factor scores are calculated as follow:
            
                                             Today
            |_____________|_______|____|_______|
                          |     P1     |   P2  |
                                  |     P3     |
            |             lookback             |
            
                                             
            r1 = (#upside - #downside) / len(P1) in P1
            
            r2 = cumret(P2)
            
            r3 = (#upside - #downside) / len(P3) in P3
            
            corr = corr(r1, r2) in lookback
            
            alpha = r3 * corr
            
        Parameters
        ----------
        short : integer, length of period used to calculte cumulate returns
        
        long : integer, length of period used to calculte upside probabilities
        
        lookback : integer, length of lookback period that used to calculate
                   correlation coefficients
                   
        Notes
        -----
            - len(P1) is equal to len(P3) to avoid bias from different length.
            
            - When rolling the window in lookback period, only stocks with
              valid data more than half of # of windows can have scores.
            
            - All invalid value will be set to 0 when calculating the scores.
        """
        if lookback < long + short:
            raise ValueError('lookback > long + short not satiesfied.')
        close = self.Data['Close'].values.T
        res = []
        x = np.zeros([close.shape[0],lookback-long-short])
        y = np.zeros([close.shape[0],lookback-long-short])
        msk = np.zeros([close.shape[0],lookback-long-short]).astype('bool')
        pb = ProgressBar()
        for i in pb(range(close.shape[1])):
            if i < (long + short):
                res.append(np.zeros(close.shape[0]))
                continue
            r1_t = np.diff(close[:,(i-long-short):(i-short)],axis=1)
            n_pos = np.sum(r1_t > 0, axis=1)
            n_neg = np.sum(r1_t < 0, axis=1)
            r1 = (n_pos - n_neg)/long
            r2 = close[:,i] / close[:,i-short] - 1
            n_r1 = np.isnan(r1)
            n_r2 = np.isnan(r2)
            r1[n_r1+n_r2] = np.nan
            r2[n_r1+n_r2] = np.nan
            x = np.c_[x[:,1:], r1]
            y = np.c_[y[:,1:], r2]
            msk = np.c_[msk[:,1:], ~n_r1*~n_r2]
            if i < lookback:
                res.append(np.zeros(close.shape[0]))
                continue
            else:
                n = np.sum(msk, axis=1)
                outlier = n < (lookback-long-short)//2
                ded = (np.nansum(x*y, axis=1)*n -
                       np.nansum(x, axis=1)*np.nansum(y, axis=1))
                dor = np.sqrt((n*np.nansum(x*x, axis=1) -
                               np.nansum(x, axis=1)**2)*
                              (n*np.nansum(y*y, axis=1) -
                               np.nansum(y, axis=1)**2))
                corr = ded / dor
                corr[np.isinf(corr)+np.isnan(corr)+outlier] = 0
                r3_t = np.diff(close[:,(i-long+1):(i+1)],axis=1)
                n_pos = np.sum(r3_t > 0, axis=1)
                n_neg = np.sum(r3_t < 0, axis=1)
                r3 = (n_pos - n_neg) / long
                res.append(corr*r3)
        alpha = np.array(res)
        return pd.DataFrame(alpha, index=self.Dates, columns=self.Tickers)
                          
    @Output()
    def Factor_std(self, window):
        """
        Purely betting on stocks with lowest standard deviation.
        
        Parameters
        ----------
        window : integer, input parameter for std
        """
        std = self.std(window)
        alpha = 1/(1+std)
        return alpha
    
    @Output()
    def Factor_std2(self, window, rolling):
        """
        Betting on oversold stocks with relatively low volatility recently.
        
        Parameters
        ----------
        window : integer, input parameter for std
        
        rolling : length of upside trend period
        """
        std = self.std(window)
        std = std.sub(std.min(axis=1),axis=0).div(std.max(axis=1).sub(
                std.min(axis=1),axis=0),axis=0)
        ret_today = self.Data['Close'].pct_change(axis=0).fillna(0)
        ret_yesterday = ret_today.shift(axis=0).fillna(0)
        
        up = ((ret_today>=ret_yesterday)*ret_today).rolling(rolling,
             axis=0).mean()
        
        up = up.sub(up.min(axis=1),axis=0).div(up.max(axis=1).sub(
                up.min(axis=1),axis=0),axis=0)
        alpha = 1/(1+std+up)
        return alpha
    
    @Output()
    def Factor_WPO(self, rolling):
        """
        Wave Period Oscillator uses wave to describe the power change during
        buy and sell periods. The indicator is calculated following
        
            theta = 2*pi / arcsin(N-day high / close)
            
            T = I(close(i) - close(i-1)) * theta
            
            wpo = EMA(T, rolling)
        
        Parameters
        ----------
        rolling : integer, parameter to smooth down WPO
        
        Notes
        -----
            - This factor may have quite high correlations with factors which
              bet on extreme reversal.
            
            - Smaller value of parameter rolling will lead to sharper jump.
              Larger value of parameter rolling will lead to later exist from
              a not bad return. Values from 3 to 22 will be reasonable.
        """
        close = self.Data['Close']
        high = self.Data['High'].rolling(3, axis=0).max()
        wpo = 4*np.pi/np.arcsin(close / high)*((close.diff(axis=0)>0)-0.5)
        wpo = wpo.ewm(span=rolling, axis=0).mean()
        alpha = -wpo
        return alpha
    
    @Output()
    def Factor_RelRev(self, window, rolling):
        """
        A factor indicating reversal effect based on positions of stocks.
        
        Parameters
        ----------
        window : integer, length of period of time needs to be compared
        """
        close = self.Data['Close']
        zscore = close.sub(close.mean(axis=1),
                           axis=0).div(close.std(axis=1),axis=0)
        diff = -zscore.diff(window, axis=0) 
        diff[np.isnan(diff) & np.isinf(diff)] = 0
        alpha = diff.ewm(span=rolling, axis=0).mean()
        return alpha
    
    @Output()
    def Factor_NightGap(self, rolling):
        return ((self.Data['High'] -
                 self.Data['Low']).rolling(rolling,axis=0).mean() /
                (self.Data['Open'].shift(-1, axis=0) -
                 self.Data['Close']).abs().rolling(rolling,axis=0).mean())
    