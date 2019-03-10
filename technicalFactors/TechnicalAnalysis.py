# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:21:23 2019

@author: Brian

"""

import pandas as pd
import numpy as np
from scipy import io
import os
from progressbar import ProgressBar
import joblib

class RollingWindow:
    """
    This is a basic class which provides functions to apply methods to rolling
    windows. Though functionality is similar to pandas.DataFrame.rolling, it
    is designed to solve problems in pandas.DataFrame.rolling.
    
    For pandas method, it can just apply functions to one column, with an
    advantage of high speed. To apply function to multiple columns, one needs
    to pass an index column to pandas method and call other columns in function
    passed, which actually will cost much time because of calling columns
    every rolling window.
    
    So in this class, when provided function is based on one column, it will
    choose pandas method to realize rolling applying. And for multiple columns
    conditions, it will simply roll the window by looping, which costs less
    time to pandas method under most conditions.
    
    Attributes
    ----------
    dataframe : a built-in dataframe which has been forward filled and then
                backward filled. Positions of null values will be stored in
                a masking list.
    
    colname : for speed reason, rolling method is processed with numpy array
              type, and columns will be stored in this attribute.
             
    rowname : same reason as illustrated above, index will be stored in this
              attribute.
              
    mask : this attribute has same length as index, which is a null indicator
           showing whether the row containing null values.
    
    arr : array type of the filled dataframe.
    
    skip : this attribute is the calculated based on mask, which indicates
           rows to be skipped based on self-defined rules.
    
    window : size of the rolling window.
    
    min_periods : minimum number of valid values in window required for
                  applying function, returning NA if not meeting the minimum
                  number. For this version, min_periods is set to be int(0.75*
                  window)    
    """
    def __init__(self, df, window):
        """
        Check whether input values are valid, and initilize attributes with
        dataframe input.
        
        Parameters
        ----------
        df : pandas dataframe, null values will be forward filled and then
             backward filled.
             
        window : positive integer, indicating size of the rolling window.
        """
        # Error control
        if not isinstance(df, pd.DataFrame):
            raise Warning('Input object type should be pandas.DataFrame!')
        
        if not isinstance(window, int) and not window > 0:
            raise TypeError('Invalid input for "window"!')
        
        # Pre-clean DataFrame
        self.dataframe = df.ffill().bfill()
        self.colname = df.columns.tolist()
        self.rowname = df.index.tolist()
        self.mask = ~(np.isnan(df.values).any(axis=1))
        if 'volume' in self.colname:
            self.dataframe.loc[~self.mask, 'volume'] = 0.0
        self.arr = self.dataframe.values
        
        # Find ineffective days
        min_periods = int(0.75*window)
        flag = (~self.mask)*1
        for i in range(window-1):
            flag += np.roll(flag, 1)
            if i == int(window*0.25):
                flag1 = flag > 0
        flag2 = flag > (window - min_periods)
        self.skip = flag1*flag2
        self.skip[:(window-1)] = np.nan
        self.window = window
        self.min_periods = min_periods
    
    def apply(self, func, col=None, *args, **kwargs):
        """
        This function automatically alters between pandas rolling method and
        self-defined method to realize rolling method. When parameter col is
        specified, it will use pandas built-in method. 
        
        Parameters
        ----------
        func : functions to be applied to a window, arguments can be followed
               after default parameters.
        
        col : columns to be applied to, default None. If specified, func
              will be applied to that column using pandas rolling method. If
              not specified, func will be applied to 2-dimention window with
              all columns inside.
        
        Notes
        -----
            - Comments below are some other version of apply methods, which
              have similar funcionality.
            
            - For pandas method or col specified, parameters for func should
              be passed in dictionary with keys indicating names of parameters.
              For self-defined method or col not specified, parameters for
              func can be directly specified.
        """
        # Input col should be name of a column
        if col is None:
            res = [np.nan]*(self.window-1)
            for i in range(self.window-1, len(self.rowname)):
                if self.skip[i]:
                    res.append(np.nan)
                    continue
                temp = self.arr[(i-self.window+1):(i+1), :]
                res.append(func(temp, *args, **kwargs))
        else:
            res = self.dataframe[col].rolling(window = self.window,
                                              min_periods = self.min_periods)
            res = res.apply(func, *args, **kwargs)
            res[self.skip] = np.nan
        return res
    
#    def rolling(self, col=None, window=5, min_periods=0):
#        # Error control
#        if window < 0:
#            raise Warning('"window" should not be negative!')
#        if min_periods > window:
#            raise Warning('"min_periods" should not be larger than window!')
#        if min_periods < 0:
#            raise Warning('"min_periods" should not be negative!')
#        
#        if col is None:
#            col = self.colname
#        
#        self.slices = []
#        for i in range(window-1, len(self.rowname)):
#            temp = self.arr[(i-window+1):(i+1),:]
#            flag = self.mask[(i-window+1):(i+1)]
#            if np.sum(flag) <= min_periods:
#                self.slices.append(None)
#                continue
#            if ~(flag[:int(window*0.25)].any()):
#                self.slices.append(None)
#                continue
#            self.slices.append(np.array(temp))
#    
#    def _apply(self, func, col=None, window=5, min_periods=0,
#               *args, **kwargs):
#        if col is None:
#            col = self.colname
#        res= []
#        for piece in self.slices:
#            if piece is None:
#                res.append(np.nan)
#                continue
#            temp = piece[:,[self.colname.index(x) for x in col]]
#            res.append(func(temp, *args, **kwargs))
##        res = pd.Series(res, index=self.rowname)
#        return res
#    
#    def apply(self, func, col=None, window=5, min_periods=0,
#              *args, **kwargs):
#        if col is not None:
#            temp = self.dataframe[col].rolling(window=window,
#                                               min_periods=min_periods)
#            res = temp.apply(func, *args, **kwargs)
#            res[~self.mask] = np.nan
#            return res
#        if (col, window, min_periods) != (self.col,
#           self.window, self.min_periods):
#            self.rolling(col = col, window = window, min_periods = min_periods)
#        return self._apply(func, col, window, min_periods, *args, **kwargs)


class TA_Indicator(RollingWindow):
    """
    This class an inherited class of RollingWindow, with same initializing
    input. Most of technical analysis indicators have been scaled and will not
    be affected by stocks' prices or market values.
    
    Attributes
    ----------
    h, l, c, v, fmv : positions of columns high_price, low_price, close_price,
                      volume, fmv, respectively. Positions are stored as
                      attributes avoiding searching in every windows.
    
    high, low, close, volume, freeMV : important values to perform technical
                                       analysis.
    """
    def __init__(self, df, window):
        """
        Initializing frequently used attributes.
        
        Parameters
        ----------
        df : pandas dataframe, which will be used to initialize RollingWindow
             class
        """
        super().__init__(df, window)
        # Pre-set frequently used indices and columns
        self.h = self.colname.index('high_price')
        self.l = self.colname.index('low_price')
        self.c = self.colname.index('close_price')
        self.v = self.colname.index('volume')
        self.fmv = self.colname.index('fmv')
        
        self.high = self.arr[:,self.h]
        self.low = self.arr[:,self.l]
        self.close = self.arr[:,self.c]
        self.volume = self.arr[:,self.v]
        self.freeMV = self.arr[:,self.fmv]
        
    #Tools
    @staticmethod
    def _ExpMultiplyer(n, alpha=None):
        multiplyer = alpha*np.array([(1-alpha)**p for p in range(n-1, -1, -1)])
        return multiplyer / np.sum(multiplyer)
    
    def _SkipNoTrading(self, res):
        if not isinstance(res, pd.Series):
            res = np.array(res)
            res[self.skip] = np.nan
            return res.tolist()
        else:
            res[self.skip] = np.nan
            return res
    
    # Functions used within one window period
    @classmethod
    def _EMA(cls, series, alpha=0.9):
        temp = series[~np.isnan(series)]
        return np.sum(temp * cls._ExpMultiplyer(len(temp), alpha))
    
    def _TR(self, arr):
        return max(arr[-1,self.h]-arr[-1,self.l],
                   arr[-1,self.h]-arr[0,self.c],
                   arr[-1,self.l]-arr[0,self.c])
    
    def _ATR(self, arr):
        TR = [np.nan]
        for i in range(arr.shape[0]-1):
            TR.append(self._TR(arr[i:(i+2),:]))
        return self._EMA(np.array(TR), alpha = 1.0 - 1.0/arr.shape[0])
    
    def _STOK(self, arr):
        H = np.max(arr[:,self.h])
        L = np.min(arr[:,self.l])
        return 100*(arr[-1,self.c]-L)/(H-L)
        
    # Rolling window with apply or pandas rolling methods
    def SMA(self):
        """
        Simple Moving Average
        """
        temp = self.dataframe['close_price'].rolling(self.window).mean()
        return self._SkipNoTrading(temp)
    
    def ROC(self):
        """
        Rate of Change
        
        This indicator is calculated by
        
            ROC = close(today)/close(N days ago) + 1
        
        """
        temp = self.dataframe['close_price'].pct_change(self.window)
        return self._SkipNoTrading(temp)
    
    def std(self):
        """
        Standard Deviation
        """
        temp = self.dataframe['close_price'].rolling(self.window).std()
        return temp.tolist()
    
    def EMA(self, alpha=None):
        """
        Exponential Moving Average
        
        Parameters
        ----------
        alpha : parameter for exponential weight, the exponential weighted
                mean will be calculated as:
                    
                    alpha*[(1-alpha)**0 * x0 + (1-alpha)**1 * x1 + ...]
                    
        """
        alpha = 2.0/(self.window + 1.0) if alpha is None else alpha
        assert alpha <= 1 and alpha >= 0, '0 <= alpha <= 1 not satisfied!'
        temp = pd.Series(self.close).ewm(alpha=alpha).mean()
        return self._SkipNoTrading(temp)
    
    def EMA1(self, alpha=None):
        """
        Exponential Moving Average (numpy version, with self-defined method)
        
        Parameters
        ----------
        alpha : parameter for exponential weight, the exponential weighted
                mean will be calculated as:
                    
                    alpha*[(1-alpha)**0 * x0 + (1-alpha)**1 * x1 + ...]
                    
        """
        alpha = 2.0/(self.window + 1.0) if alpha is None else alpha
        assert alpha <= 1 and alpha >= 0, '0 <= alpha <= 1 not satisfied!' 
        return super().apply(self._EMA, 
                             col='close_price',
                             kwargs={'alpha':alpha}).tolist()
    
    def MOM(self):
        """
        Momentum
        
        To eliminate effects casted by scale, it is calculate in ratio form,
        which equals to 1 + ROC.
        
        This indicator is calculated by
            
            MOM = close(today) / close(N days ago)
            
        """
        return (np.array(self.ROC()) + 1.0).tolist()
    
    def BBands(self):
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
        middle = np.array(self.SMA())
        std = np.array(self.std())
        upper = middle + 2*std
        lower = middle - 2*std
        
        return [self._SkipNoTrading(upper),
                self._SkipNoTrading(middle),
                self._SkipNoTrading(lower)]
    
    def ATR(self):
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
        return super().apply(self._ATR)
    
    def PPSR(self):
        """
        Pivot Points, Resistances and Supports
        
        This indicator is calculated by
        
            PP = (high + low + close) / 3
            
            R1 = 2 * PP - low
            S1 = 2 * PP - high
            
            R2 = PP + high - low
            S2 = PP - high + low
            
            R3 = high + 2 * (PP - low)
            S3 = low - 2 * (high - PP)
        
        Returns
        -------
        [S3, S2, S1, PP, R1, R2, R3] : 6 lists in a list, which are Pivot
                                       Points, S level 1 2 3, R level 1 2 3,
                                       respectively.
        
        Notes
        -----
            - Today'indicators are generated from yesterday's prices and will
              be fixed for the whole day. And this week's indicators are 
              generated from last week's prices and will be fixed for entire
              week.
              
            - The second support and resistance levels can be used to identify
              overbought and oversold situations.
        """
        period = 22
        high = self.dataframe.high_price.rolling(period).max().values
        low = self.dataframe.low_price.rolling(period).min().values
        close = self.close
        PP = (high + low + close) / 3
        R1 = 2 * PP - low
        S1 = 2 * PP - high
        R2 = PP + high - low
        S2 = PP - high + low
        R3 = high + 2*(PP - low)
        S3 = low - 2*(high - PP)
        return [self._SkipNoTrading(S3),
                self._SkipNoTrading(S2),
                self._SkipNoTrading(S1),
                self._SkipNoTrading(PP),
                self._SkipNoTrading(R1),
                self._SkipNoTrading(R2), 
                self._SkipNoTrading(R3)]

    def STO(self):
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
        STOK = super().apply(self._STOK)
        STOD = pd.Series(STOK).rolling(3).mean()
        return [STOK, self._SkipNoTrading(STOD)]
    
    def TRIX(self):
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
        smooth1 = pd.Series(self.close).ewm(span=self.window).mean()
        smooth2 = smooth1.ewm(span=self.window).mean()
        smooth3 = smooth2.ewm(span=self.window).mean()
        temp = smooth3.pct_change()
        return self._SkipNoTrading(temp)
    
    def ADX(self, nadx):
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
        up = np.hstack([np.nan, self.high[1:] - self.high[:-1]])
        down = np.hstack([np.nan, self.low[:-1] - self.low[1:]])

        dm = [[up[i] if up[i]>down[i] and up[i]>0 else 0,
               down[i] if down[i]>up[i] and down[i]>0 else 0]
              for i in range(len(up))]
        
        ATR = pd.Series(self.ATR())
        dm = 100.0*pd.DataFrame(dm).ewm(span=self.window,
                               axis=1).mean().divide(ATR, axis=0)
        
        DX = dm.diff(1,axis=1).iloc[:,-1].abs() / dm.sum(axis=1)
       
        ADX = 100.0 * DX.ewm(span=14).mean()
        return [self._SkipNoTrading(ADX),
                self._SkipNoTrading(dm.iloc[:,0]),
                self._SkipNoTrading(dm.iloc[:,1])]
    
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
        EMAshort = self.dataframe['close_price'].ewm(span=n_short).mean()
        EMAlong = self.dataframe['close_price'].ewm(span=n_long).mean()
        
        DIF = EMAshort - EMAlong
        DEM = DIF.ewm(span=9).mean()
        OSC = DIF - DEM
        
        return [self._SkipNoTrading(DIF),
                self._SkipNoTrading(DEM),
                self._SkipNoTrading(OSC)]
    
    def RSI(self):
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
        ret = np.hstack([0.0, np.diff(self.close)])
        ud = [[ret[i],0] if ret[i]>=0 else [0,-ret[i]] for i in range(len(ret))]
        
        RS = pd.DataFrame(ud).ewm(span=self.window).mean()
        RS = RS.iloc[:,0] / RS.iloc[:,1]
        RSI = (1 - 1/(1+RS)).values*100
        
        return self._SkipNoTrading(RSI)
    
    def OBV(self):
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
        OBV = (np.hstack([0, 2*((self.close[1:] >
                                self.close[:-1])-0.5)])*
              self.volume/self.freeMV*10000)
        OBV = pd.Series(OBV).rolling(self.window).mean()
        return self._SkipNoTrading(OBV)

class TechnicalAnalysis(TA_Indicator, RollingWindow):
    """
    This class is an inherited class of TA_Indicator, which uses indicator
    functions to perform technical analysis.
    """
    def __init__(self, arr, dates, tickers, prices, f_dir = './Factors',
                 parallel = True, smooth = True):
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
        
        f_dir : directory to output factors.
        """
        self.large_arr = arr
        self.dates = pd.Series(dates)
        self.dates.name = 'Date'
        self.tickers = tickers
        self.cname = prices + ['volume']
        
        self.func_map = {'SMA': super().SMA,
                         'EMA': super().EMA,
                         'ATR': super().ATR,
                         'ADX': super().ADX,
                         'ROC': super().ROC,
                         'std': super().std,
                         'BBands': super().BBands,
                         'MACD': super().MACD,
                         'OBV': super().OBV,
                         'MOM': super().MOM,
                         'PPSR': super().PPSR,
                         'RSI': super().RSI,
                         'STO': super().STO,
                         'TRIX': super().TRIX}
        
        self.indicators = {}
        
        self.large_arr = np.append(arr, [arr[prices.index('turn'),:,:]*
                                   arr[prices.index('mv'),:,:]/
                                   arr[prices.index('close_price'),:,:]/100],0)
        self.large_arr[:,arr[prices.index('turn'),:,:]==0] = np.nan
        
        self.f_dir = f_dir
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)
            
        self.c = self.cname.index('close_price')
        self.h = self.cname.index('high_price')
        self.l = self.cname.index('low_price')
        self.v = self.cname.index('volume')
        
        self.parallel = parallel
        self.smooth = smooth
    
    def Output(self, res, fname):
        """
        Output given factor under ./f_dir/ using given file name.
        
        Parameters
        ----------
        res : factor table to be output
        
        fname : file name for the output file
        
        Notes
        -----
            - For current stage, factors will be output as .csv files with
              header and dates.
              
            - Smoothing method used for output is exponential mean with span
              = 5.
        """
        temp = res.T
        if not isinstance(temp, pd.DataFrame):
            temp = pd.DataFrame(np.array(temp), index=self.dates,
                     columns=self.tickers)
        else:
            temp.index = self.dates
            temp.columns = self.tickers
        temp.to_csv(f'{self.f_dir}/{fname}.csv')
        if self.smooth:
            temp.ewm(span=5, 
                     axis=0).mean().to_csv(f'{self.f_dir}/{fname}_em.csv')
        print(f'{fname} Done!')
        return temp
    
    def SliceGenerator(self):
        """
        Dataframe generator which yields dataframe of tickers. It is a
        iterable generator, so please put this function to iterator in order
        to produce the slices.
        """
        for idx in range(len(self.tickers)):
            yield pd.DataFrame(self.large_arr[:,idx,:].T,
                            index=pd.to_datetime(self.dates),
                            columns=self.cname)
    

    def IndicatorGenerator(self, indicator, window=10, *args, **kwargs):
        """
        By specifying name of indicator, this function can generate indicators
        in target window. Since not all indicators need the window setting,
        this parameter will not have impact on those indicators.
        
        Parameters
        ----------        
        indicator : string, name of the indicator returned.
        
        window : integer, indicating size of the rolling window.
        
        Notes
        -----
            - Additional parameters can be directly input along with names.
        """
        for df in self.SliceGenerator():
            super().__init__(df, window)
            yield self.func_map[indicator](*args, **kwargs)
    
    def GetIndicator_all(self, indicator, window=10, *args, **kwargs):
        """
        This function returned the specified indicator for all tickers.
        
        Parameters
        ----------
        indicator : string, name of the indicator returned.
        
        window : integer, indicating size of the rolling window.
        
        Notes
        -----
            - Additional parameters can be directly input along with names.
            
            - This function mainly designed for indicators that can be used
              apart from prices.
              
            - Parallel is realized using joblib with default backend loky.
            
            - # of cores used for parallel is set to use all cores available.
        """
        if self.parallel:
            df_list = []
            for df in self.SliceGenerator():
                df_list.append(df)
        
            def worker(df):
                temp = TA_Indicator(df, window)
                return getattr(temp, indicator)(*args, **kwargs)
    
            res = joblib.Parallel(n_jobs=-1, 
                                  backend='loky')(joblib.delayed(worker)(x)
                                  for x in df_list)
        else:
            res = []
            for indicator in self.IndicatorGenerator(indicator, window,
                                                     *args, **kwargs):
                res.append(indicator)
                
        return np.array(res)
    
    def GetIndicator(self, idx, indicator, window=10, *args, **kwargs):
        """
        This function is defined for testing usage, which returns specified
        indicator of target ticker.
        
        Parameters
        ----------
        idx : index of a ticker in self.tickers.
        
        indicator : string, name of the indicator returned.
        
        window : integer, indicating size of the rolling window.
        
        Notes
        -----
            - Additional parameters can be directly input along with names.
            
            - Since some indicator should be used along with prices or volume,
              the dataframe slice will also be returned in this function.
        """
        temp = pd.DataFrame(self.large_arr[:,idx,:].T,
                            index=pd.to_datetime(self.dates),
                            columns=self.cname)
        super().__init__(temp, window)
        return temp, self.func_map[indicator](*args, **kwargs)
    
    def PPSR_band(self, idx):
        """
        This function serves for PPSR indicator.
        
        By lagging serveral days, prices will cross over bands. And this
        function will record the position of price.
        
        Parameters
        ----------
        idx : index of a ticker in self.tickers.
        
        Notes
        -----
            - Lagging period is now set to 10, which is also a parameter need
              to be fine tuned.
              
            - If the indicator is nan, then position returned is also nan.
        """
        df, ind = self.GetIndicator(idx, 'PPSR', 10)
        lag = 10
        ind = np.roll(np.array(ind), shift=lag, axis=-1)
        ind[:,:lag] = np.nan
        mask = np.isnan(ind).any(axis=0)
        price = df.close_price.values
        flag = (np.sum(price.reshape([1,-1])>ind,axis=0)).astype('float16')
        flag[mask] = np.nan
        return flag
    
    @staticmethod
    def PatternScore(flag, now, up, down, updown, downup, patience):
        """
        This function calculates probabilities for conditions of up trend,
        down trend, reversal down trend after up trend, reversal up trend
        after down trend, from today's level.
        
        Parameters
        ----------
        flag : an one dimention array recording status of prices, positions
               exactly.
               
        now : integer, status of today's price.
        
        up : integer, defines # of levels being crossovered can be regarded
             as an up trend breakthrough.
        
        down : integer, defines # of levels being crossovered can be regarded
               as a down trend breakthrough.
               
        updown : integer, defines # of levels being crossovered can be
                 regarded as a reversal down trend after the up trend.
                 
        downup : integer, defines # of levels being crossovered can be
                 regarded as a reversal up trend after the down trend.
                 
        patience : integer, length of period that can be allowed to detect
                   a trend.
        
        Returns
        -------
        [P_up - P_down, P_up*(1 - P_updown) - P_down*(1 - P_downup)]
        : 2 probability differences in a list.
        Notes
        -----
            - P_updown and P_downup are calculated as conditional probability.
            
            - Breakthrough points are detected by finding all condition
              satiesfied points within patience period, and then add the first
              point to set. This algorithm gets correct results but still has
              optimization possibility in time cost.
        """ 
        # Dealing with too much missing values
        if np.sum(np.isnan(flag)) > len(flag) * 0.5:
            return np.nan
        
        flat_set = set()
        up_found = set()
#        down_found = set()
#        updown_found = set()
#        downup_found = set()
        
        diff = np.r_[np.nan, np.diff(flag)]
        flat_set.update(np.where(diff == 0)[0])
        flat_set.update(np.where(np.isnan(diff))[0])
        
        now_found = np.where(flag == now)[0]
        if len(now_found) == 0:
            return np.nan
#            return [np.nan] * 2
        
        for idx in now_found[:-patience]:
            temp_idx = np.arange(idx, idx+patience)
            temp_flag = flag[temp_idx] >= (now + up)
            if temp_flag.any():
                if temp_idx[temp_flag][0] not in flat_set:
                    up_found.add(temp_idx[temp_flag][0])
#            temp_flag = flag[temp_idx] <= (now + down)
#            if temp_flag.any():
#                if len([1 for x in temp_idx[temp_flag]
#                    if x in flat_set]) == 0:
#                    down_found.add(temp_idx[temp_flag][0])

#        for idx in up_found:
#            if idx + patience > len(flag):
#                continue
#            temp_idx = np.arange(idx, idx+patience)
#            temp_flag = flag[temp_idx] <= (now + up + updown)
#            if temp_flag.any():
#                if temp_idx[temp_flag][0] not in flat_set:
#                    updown_found.add(temp_idx[temp_flag][0])
#        
#        for idx in down_found:
#            if idx + patience > len(flag):
#                continue
#            temp_idx = np.arange(idx, idx+patience)
#            temp_flag = flag[temp_idx] >= (now + down + downup)
#            if temp_flag.any():
#                if temp_idx[temp_flag][0] not in flat_set:
#                    downup_found.add(temp_idx[temp_flag][0])
                
        P_up = len(up_found) / len(now_found)
#        P_updown = (len(updown_found) / len(up_found)
#                    if len(up_found) != 0 else 0)
#        P_down = len(down_found) / len(now_found)
#        P_downup = (len(downup_found) / len(down_found)
#                    if len(down_found) != 0 else 0)
#        
        return P_up
#        return [P_up - P_down,
#                P_up*(1 - P_updown) - P_down*(1 - P_downup)]
    
    @staticmethod
    def PatternScore1(flag, now, up, updown, patience):
        """
        This function calculates probabilities for conditions of up trend,
        down trend, reversal down trend after up trend, reversal up trend
        after down trend, from today's level.
        
        Parameters
        ----------
        flag : an one dimention array recording status of prices, positions
               exactly.
               
        now : integer, status of today's price.
        
        up : integer, defines # of levels being crossovered can be regarded
             as an up trend breakthrough.
        
        down : integer, defines # of levels being crossovered can be regarded
               as a down trend breakthrough.
               
        updown : integer, defines # of levels being crossovered can be
                 regarded as a reversal down trend after the up trend.
                 
        downup : integer, defines # of levels being crossovered can be
                 regarded as a reversal up trend after the down trend.
                 
        patience : integer, length of period that can be allowed to detect
                   a trend.
        
        Returns
        -------
        [P_up - P_down, P_up*(1 - P_updown) - P_down*(1 - P_downup)]
        : 2 probability differences in a list.
        Notes
        -----
            - P_updown and P_downup are calculated as conditional probability.
            
            - Breakthrough points are detected by finding all condition
              satiesfied points within patience period, and then add the first
              point to set. This algorithm gets correct results but still has
              optimization possibility in time cost.
        """ 
        # Dealing with too much missing values
        if np.sum(np.isnan(flag)) > len(flag) * 0.5:
            return np.nan
        
        flat_set = set()
#        down_found = set()
#        updown_found = set()
#        downup_found = set()
        
        diff = np.r_[np.nan, np.diff(flag)]
        flat_set.update(np.where(diff == 0)[0])
        flat_set.update(np.where(np.isnan(diff))[0])
        
        now_found = np.where(flag == now)[0]
        if len(now_found) == 0:
            return np.nan
#            return [np.nan] * 2
        
        up_count = 0
        for idx in now_found[:-patience]:
            temp_idx = np.arange(idx, idx+patience)
            up_flag = flag[temp_idx] >= (now + up)
            if up_flag.any():
                updown_flag = flag[temp_idx]<=(now + up + updown)
                if not (temp_idx[up_flag][0]<temp_idx[updown_flag]).any():
                    up_count += 1
            else:
                up_flag = flag[temp_idx] >= (now + up - 1)
                if up_flag.any():
                    updown_flag = flag[temp_idx]<=(now + up + updown)
                    if not (temp_idx[up_flag][0]<temp_idx[updown_flag]).any():
                        up_count += 0.5
#            temp_flag = flag[temp_idx] <= (now + down)
#            if temp_flag.any():
#                if len([1 for x in temp_idx[temp_flag]
#                    if x in flat_set]) == 0:
#                    down_found.add(temp_idx[temp_flag][0])

#        for idx in up_found:
#            if idx + patience > len(flag):
#                continue
#            temp_idx = np.arange(idx, idx+patience)
#            temp_flag = flag[temp_idx] <= (now + up + updown)
#            if temp_flag.any():
#                if temp_idx[temp_flag][0] not in flat_set:
#                    updown_found.add(temp_idx[temp_flag][0])
#        
#        for idx in down_found:
#            if idx + patience > len(flag):
#                continue
#            temp_idx = np.arange(idx, idx+patience)
#            temp_flag = flag[temp_idx] >= (now + down + downup)
#            if temp_flag.any():
#                if temp_idx[temp_flag][0] not in flat_set:
#                    downup_found.add(temp_idx[temp_flag][0])
                
        P_up = up_count / len(now_found)
#        P_updown = (len(updown_found) / len(up_found)
#                    if len(up_found) != 0 else 0)
#        P_down = len(down_found) / len(now_found)
#        P_downup = (len(downup_found) / len(down_found)
#                    if len(down_found) != 0 else 0)
#        
        return P_up
#        return [P_up - P_down,
#                P_up*(1 - P_updown) - P_down*(1 - P_downup)]
        
    
    def PPSR_factor(self, idx, lookback):
        """
        Rolling 3-month window to call self.PatternScore.
        
        Parameters
        ----------
        idx : index of a ticker in self.tickers.
        """
        flag = self.PPSR_band(idx)
        factor = []
        for i in range(len(flag)):
            if i >= lookback:
                temp_flag = flag[(i-lookback):i]
#                factor.append(self.PatternScore(temp_flag,
#                                                flag[i], 2, -1, -2, 2, 10))
                factor.append(self.PatternScore1(temp_flag,
                                                 flag[i], 2, -2, 5))
            else:
                factor.append(np.nan)
#                factor.append([np.nan]*2)
        return factor
    
    def PPSR_f_generator(self, lookback):
        """
        PPSR factor generator.
        """
        for idx in range(len(self.tickers)):
            yield self.PPSR_factor(idx, lookback)
    
    def Factor_PPSR(self, lookback):
        """
        This function uses self.PPSR_f_generator to produce factors.
        """
        res = []
        for factor in self.PPSR_f_generator(lookback):
            res.append(factor)
        
        return self.Output(res, f'PPSR_{int(lookback)}')
    
    def Factor_MADMOM(self, lookback, short, long):
        """
        This factor is designed based on trend and momentum.
        
        Parameters
        ----------
        lookback : integer, length of period for detecting trends.
        
        short : integer, length of short term spread.
        
        long : integer, length of long term spread.
        
        Notes
        -----
            - This factor uses z-score as factor scores, which will be 
              affected by outliers.
        """
        ret_today = np.c_[[np.nan]*self.large_arr.shape[1],
                          self.large_arr[self.c,:,1:] -
                          self.large_arr[self.c,:,:-1]]
        ret_yesterday = pd.DataFrame(np.c_[[np.nan]*ret_today.shape[0], 
                                           ret_today[:,:-1]]).fillna(0)
        ret_today = pd.DataFrame(ret_today).fillna(0)
        
        up = ((ret_today>=ret_yesterday)*(ret_today-
              ret_yesterday)).rolling(lookback,axis=1).sum()
        down = ((ret_today<ret_yesterday)*(ret_today-
                ret_yesterday)).rolling(lookback,axis=1).sum()
        
        spread_allday = pd.DataFrame(self.large_arr[
                        self.h,:,:]
                        - self.large_arr[self.l,:,:])
        
        mad = (spread_allday.rolling(short,axis=1).mean() /
               spread_allday.rolling(long,axis=1).mean())
        
        temp = (up-down)/(up+down)
        addit = ((temp - temp.rolling(10,axis=1).mean())/
                 temp.rolling(10,axis=1).std())
        
        alpha = ((mad - mad.rolling(10,axis=1).mean())/
                 mad.rolling(10,axis=1).std())
        
        up[pd.isnull(up)] = 0
        
        alpha = up**3 * (alpha + addit > 2.5)
        
        alpha[pd.isnull(alpha)] = 0
        alpha = alpha.ewm(alpha=0.8,axis=1).mean()
        
        return self.Output(alpha,
                    f'MADMOM_{int(lookback)}_{int(short)}_{int(long)}')
    
    def Factor_RSI_bot(self, short, long, rolling):
        """
        Summation of distances of short-term RSI from long-term RSI, 
        30 and minimal value among rolling period.
        
        Parameters
        ----------
        short : integer, length for short term RSI period.
        
        long : integer, length for long term RSI period.
        
        rolling : integer, length for minimal value period.
        """
        ind_short = pd.DataFrame(self.GetIndicator_all('RSI', window=short))
        ind_long = pd.DataFrame(self.GetIndicator_all('RSI', window=long))
        
        temp_short = ind_short.fillna(100)
        temp_long = ind_long.fillna(100)
        
        support = temp_short.rolling(rolling, axis=1).min()
        alpha = (30-temp_short)/(1-support+temp_short)
        return self.Output(alpha,
                           f'RSI_bot_{int(short)}_{int(long)}_{int(rolling)}')

    
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
        ind = pd.DataFrame(self.GetIndicator_all('RSI', window=window))
        alpha = 100 - ind
        alpha[~(alpha >= 70)] = 0
        alpha = alpha.ewm(span = 5).mean()
        
        return self.Output(alpha, f'RSI_low_{int(window)}')
    
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
        ind_short = self.GetIndicator_all('MACD', n_short=ss, n_long=sl)
        ind = self.GetIndicator_all('MACD', n_short=ls, n_long=ll)
        
        temp_short = pd.DataFrame(ind_short[:,2,:]).fillna(0)
        temp_long = pd.DataFrame(ind[:,2,:]).fillna(0)
        
        support = temp_short.rolling(rolling,axis=1).min()
        alpha = support + temp_long - 2*temp_short
        
        return self.Output(alpha,
               f'MACD_{int(ss)}_{int(sl)}_{int(ls)}_{int(ll)}_{int(rolling)}')
    
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
        ind_short = self.GetIndicator_all('TRIX', window=short)
        ind = self.GetIndicator_all('TRIX', window=long)
        support = pd.DataFrame(ind).rolling(rolling,axis=1).min()
        support_short = pd.DataFrame(ind_short).rolling(rolling,axis=1).min()
        alpha = support+support_short+ind-3*ind_short
        
        return self.Output(alpha,
                           f'TRIX_{int(short)}_{int(long)}_{int(rolling)}')
    
    def Factor_TRIX_ret(self, short, long, rolling):
        """
        Betting on extreme TRIX reversal along with large drawdown.
        
        Parameters
        ----------
        short : integer, length for short term TRIX
        
        long : integer, length for long term TRIX
        
        rolling : integer, length for minimal value period
        """
        ind_short = self.GetIndicator_all('TRIX', window=short)
        ind = self.GetIndicator_all('TRIX', window=long)
        ret = self.GetIndicator_all('ROC', window=short)
        support = pd.DataFrame(ind).rolling(rolling,axis=1).min()
        support_short = pd.DataFrame(ind_short).rolling(rolling,axis=1).min()
        support_flag = pd.DataFrame(ret).rolling(rolling,axis=1).min()
        alpha = (support+support_short+ind - 3*ind_short)/(1+support_flag-ret)
        
        return self.Output(alpha,
                           f'TRIX_ret_{int(short)}_{int(long)}_{int(rolling)}')
        
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
        ind_short = self.GetIndicator_all('TRIX', window=short)
        ind = self.GetIndicator_all('TRIX', window=long)
        vol = self.GetIndicator_all('OBV', window=short)
        s_vol = ((vol - np.nanmin(vol,axis=0))/
                 (np.nanmax(vol,axis=0)-np.nanmin(vol)))
        support = pd.DataFrame(ind).rolling(rolling,axis=1).min()
        support_short = pd.DataFrame(ind_short).rolling(rolling,axis=1).min()
        alpha = support + support_short + ind - 3*ind_short
        alpha = ((alpha-np.nanmin(alpha,axis=0))/
                 (np.nanmax(alpha,axis=0)-np.nanmin(alpha,axis=0)))
        alpha = np.exp(1 + s_vol)*alpha
        
        return self.Output(alpha,
                           f'TRIX_vol_{int(short)}_{int(long)}_{int(rolling)}')
    
    def Factor_BBands(self, short, long):
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
        ind_short = self.GetIndicator_all('BBands', window=short)
        ind = self.GetIndicator_all('BBands', window=long)
        lower = pd.DataFrame(ind[:,-1,:])
        lower_short = pd.DataFrame(ind_short[:,-1,:])
        dist = (lower - self.large_arr[self.c,:,:]).ewm(alpha=0.9, 
               axis=1).mean()
        dist_short = (lower_short - self.large_arr[self.c,:,:]).ewm(alpha=0.9,
                     axis=1).mean()
        alpha = dist + dist_short
        
        return self.Output(alpha, f'BBands_{int(short)}_{int(long)}')
    
    def Factor_ADX(self, nadx, rolling):
        """
        Betting on extreme reversal of distance between upside movement and
        downside movement, with an enhancement from trend strength.
        
        Parameters
        ----------
        nadx : integer, input parameter for ADX indicator
        
        rolling : integer, length for minimal value period
        
        Notes
        -----
            - For the reason that ADX is an indicator for trend strength, 
              sometimes the trend strengh signal may not sensitive enough
              due to its correlation to ATR indicator.
        """
        ind = self.GetIndicator_all('ADX', nadx=nadx)
        gap_DI = pd.DataFrame(ind[:,1,:] - ind[:,2,:])
        support = gap_DI.rolling(rolling,axis=1).min()
        alpha = (support - gap_DI) * (100-ind[:,0,:])
        
        return self.Output(alpha, f'ADX_{int(nadx)}_{int(rolling)}')
        
    def Factor_Reversal(self, window):
        """
        Traditional reversal factor which longs stocks with lowest returns
        last time period.
        
        Parameters
        ----------
        window : integer, input parameter for ROC
        """
        alpha = -self.GetIndicator_all('ROC', window)
        
        return self.Output(alpha, f'REV_{int(window)}')

    def Factor_MOM(self, window):
        """
        Traditional momentum factor which longs stocks having highest returns
        last time period.
        
        Parameters
        ----------
        window : integer, input parameter for MOM
        """
        alpha = self.GetIndicator_all('MOM', 5)
        
        return self.Output(alpha, f'MOM_{int(window)}')
        
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
        sma = TA.GetIndicator_all('SMA', short)
        sma_long = self.GetIndicator_all('SMA', long)
        flag = sma / sma_long
        flag2 = pd.DataFrame(flag).shift(axis=1)
        flag3 = flag2.shift(axis=1)
        alpha = (flag2 > flag)*(flag2 - flag)*(flag3 > flag2)
        alpha = alpha.ewm(span=5, axis=1).mean()
        return self.Output(alpha, f'MAD_{int(short)}_{int(long)}')
        
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
        ind = (self.GetIndicator_all('SMA',short)/
               self.GetIndicator_all('SMA',long))
        std = (self.GetIndicator_all('std',short)/
               self.GetIndicator_all('std',long))
        ind2 = pd.DataFrame(ind).shift(axis=1)
        ind3 = ind2.shift(axis=1)
        flag = std / pd.DataFrame(std).shift(axis=1)
        alpha = (ind2 > ind)*(ind2 - ind)*(ind3 > ind2)*flag
        alpha = pd.DataFrame(alpha).ewm(span=5,axis=1).mean()
        return self.Output(alpha, f'MAD_std_{int(short)}_{int(long)}')
    
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
        ind = self.GetIndicator_all('STO', window)
        flag = pd.DataFrame(ind[:,1,:] - ind[:,0,:])
        support = flag.rolling(rolling, axis=1).max()
        alpha = flag - support
        alpha = alpha.ewm(span=5, axis=1).mean()
        return self.Output(alpha, f'STO_{int(window)}_{int(rolling)}')
        
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
        ind_short = TA.GetIndicator_all('OBV', short)
        ind = TA.GetIndicator_all('OBV',long)
        support = pd.DataFrame(ind_short).rolling(rolling,axis=1).min()
        alpha = (support - ind_short)
        alpha = alpha.ewm(span=5,axis=1).mean()
        return self.Output(alpha, 
                           f'OBV_{int(short)}_{int(long)}_{int(rolling)}')
        
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
        ind_short = pd.DataFrame(self.GetIndicator_all('OBV', short))
        ind = pd.DataFrame(self.GetIndicator_all('OBV', long))
        support = ind_short.rolling(rolling,axis=1).min()
        alpha = (support - ind_short)*(ind<ind_short)*(ind.shift(axis=1)>
                 ind_short.shift(axis=1))
        alpha = alpha.ewm(span=5,axis=1).mean()
        return self.Output(alpha,
                           f'OBV2_{int(short)}_{int(long)}_{int(rolling)}')
    
    
#TA = TechnicalAnalysis(Data, Dates, Tickers, data_col)
#
## PPSR_factor = TA.Factor_PPSR()
#BBands_factor = TA.Factor_BBands()
#TRIX_vol_factor = TA.Factor_TRIX_vol()
#TRIX_ret_factor = TA.Factor_TRIX_ret()
#TRIX_factor = TA.Factor_TRIX()
#MACD_factor = TA.Factor_MACD()
#RSI_low_factor = TA.Factor_RSI_low()
#RSI_bot_factor = TA.Factor_RSI_bot()
#ret_aft_factor = TA.ret_afternoon()
#
#pd.DataFrame(RSI_factor).to_csv('./Factor/F_RSI_low.csv', header=False, index=False)
#g = np.array(PPSR_factor)
#TA.PatternScore1(TA.PPSR_band(10),4,2,-2,5)
#
#siw = TA.Factor_RSI_bot()
#
#ss = np.sum(~np.isnan(g) * (g != 0), axis=0) / g.shape[1]
#
#pd.DataFrame(g).to_csv('PPSR_4.csv', header=False, index=False)
#
#lag = 10
#ind = np.roll(np.array(g.res[16]),shift=lag,axis=-1)
#ind[:,:,:lag] = np.nan
#mask = np.isnan(ind).any(axis=(0,1))
#price = Data[data_col.index('close_price'),16,:]
#flag = np.sum(price.reshape([1,1,-1]) > ind, axis=1) - 3.5
#flag = (flag + 0.5*np.sign(flag))
#flag[:,mask] = np.nan
#np.where(flag[0,:] == -2)[0]
#
#cc = np.c_[[np.nan]*3,np.diff(flag,axis=1)]
#cc = np.nan_to_num(cc)
#
#g = TechnicalAnalysis(Data, Dates, Tickers, data_col)
#df,ind = g.GetIndicator(10,'PPSR',10)
#g = np.array(g)
#
#gg = g.arr
## Timing
#%timeit a = TechnicalAnalysis(df, window=10) # 1.53 ms
#%timeit s2 = a.ADX(nadx=5) # 165 ms
#%timeit s2 = a.ATR() # 152 ms
#%timeit s2 = a.BBands() # 1.87 ms
#%timeit s2 = a.EMA() # 0.94 ms
#%timeit s2 = a.MACD(n_short=12, n_long=24) # 4.88 ms
#%timeit s2 = a.MOM() # 0.92 ms
#%timeit s2 = a.OBV() # 0.76 ms
#%timeit s2 = a.PPSR() # 0.72 ms
#%timeit s2 = a.ROC() # 0.75 ms
#%timeit s2 = a.RSI() # 6.6 ms
#%timeit s2 = a.SMA() # 0.64 ms
#%timeit s2 = a.STO() # 34.4 ms
#%timeit s2 = a.TRIX() # 3.07 ms
#
#a.apply(a._ADX,nadx=10,ATR=)

if __name__ == '__main__':
    
    # Import data
    data_dir = './Data/'
    Dates = [str(x[0][0]) for x in io.loadmat(f'{data_dir}date.mat').popitem()[1]]
    Tickers = [str(x[0][0]) for x in io.loadmat(f'{data_dir}ticker.mat').popitem()[1]]
    
    Data = []
    data_col = []
    for data in os.listdir(data_dir):
        if 'data' not in data:
            continue
        temp = io.loadmat(f'{data_dir}{data}').popitem()
        data_col.append(temp[0])
        Data.append(temp[1].tolist())
    
    Data = np.array(Data)
    
    # 
    CSI500_ret = io.loadmat(f'{data_dir}csi500_dy_return.mat').popitem()[1].reshape([-1])
    
    price_col = [i for i in range(len(data_col)) if 'price' in data_col[i]]
    Data_ret = Data[price_col,:,1:]/Data[price_col,:,:-1] - CSI500_ret[1:]
    mask = np.isnan(Data)
    Data_ret[np.isnan(Data_ret)] = 1   
    Data_ret = np.cumprod(Data_ret, axis=2)
    Data[price_col,:,1:] = Data_ret
    
    Data[mask] = np.nan
    del Data_ret
    
    
#    [x for x in dir(TA) if callable(getattr(TA,x))]
# 'Factor_ADX',
# 'Factor_BBands',
# 'Factor_MACD',
# 'Factor_MAD',
# 'Factor_MADMOM',
# 'Factor_MAD_std',
# 'Factor_MOM',
# 'Factor_OBV',
# 'Factor_OBV2',
# 'Factor_PPSR',
# 'Factor_RSI_bot',
# 'Factor_RSI_low',
# 'Factor_Reversal',
# 'Factor_STO',
# 'Factor_TRIX',
# 'Factor_TRIX_ret',
# 'Factor_TRIX_vol',
    
    TA = TechnicalAnalysis(Data, Dates, Tickers, data_col, './Factors')
#    TA.Factor_TRIX_vol.__code__.co_varnames[1:TA.Factor_TRIX_vol.__code__.co_argcount]
    temp = TA.Factor_ADX(nadx = 5, rolling = 5)
    temp = TA.Factor_ADX(nadx = 5, rolling = 10)
    temp = TA.Factor_ADX(nadx = 5, rolling = 22)
    temp = TA.Factor_ADX(nadx = 10, rolling = 5)
    temp = TA.Factor_ADX(nadx = 10, rolling = 10)
    temp = TA.Factor_ADX(nadx = 10, rolling = 22)
    temp = TA.Factor_BBands(short = 3, long = 10)
    temp = TA.Factor_BBands(short = 3, long = 22)
    temp = TA.Factor_BBands(short = 5, long = 10)
    temp = TA.Factor_BBands(short = 5, long = 22)
    temp = TA.Factor_BBands(short = 7, long = 10)
    temp = TA.Factor_BBands(short = 7, long = 22)
    temp = TA.Factor_MACD(ss = 3, sl = 5, ls = 5, ll = 10, rolling=5)
    temp = TA.Factor_MACD(ss = 3, sl = 5, ls = 5, ll = 10, rolling=10)
    temp = TA.Factor_MACD(ss = 5, sl = 9, ls = 9, ll = 12, rolling=5)
    temp = TA.Factor_MACD(ss = 5, sl = 9, ls = 9, ll = 12, rolling=10)
    temp = TA.Factor_MACD(ss = 5, sl = 12, ls = 12, ll = 24, rolling=5)
    temp = TA.Factor_MACD(ss = 5, sl = 12, ls = 12, ll = 24, rolling=10)
    temp = TA.Factor_MACD(ss = 5, sl = 24, ls = 12, ll = 66, rolling=5)
    temp = TA.Factor_MACD(ss = 5, sl = 24, ls = 12, ll = 66, rolling=10)
    temp = TA.Factor_MACD(ss = 5, sl = 9, ls = 12, ll = 26, rolling=5)
    temp = TA.Factor_MACD(ss = 5, sl = 9, ls = 12, ll = 26, rolling=10)
    temp = TA.Factor_MAD(short = 3, long = 22)
    temp = TA.Factor_MAD(short = 3, long = 66)
    temp = TA.Factor_MAD(short = 5, long = 22)
    temp = TA.Factor_MAD(short = 5, long = 66)
    temp = TA.Factor_MAD(short = 10, long = 22)
    temp = TA.Factor_MAD(short = 10, long = 66)
    temp = TA.Factor_MAD(short = 3, long = 200)
    temp = TA.Factor_MAD(short = 10, long = 200)
    temp = TA.Factor_MAD(short = 22, long = 200)
    temp = TA.Factor_MADMOM(lookback = 10, short = 3, long = 22)
    temp = TA.Factor_MADMOM(lookback = 10, short = 10, long = 22)
    temp = TA.Factor_MADMOM(lookback = 10, short = 3, long = 200)
    temp = TA.Factor_MADMOM(lookback = 10, short = 10, long = 200)
    temp = TA.Factor_MADMOM(lookback = 10, short = 22, long = 200)
    temp = TA.Factor_MADMOM(lookback = 22, short = 3, long = 22)
    temp = TA.Factor_MADMOM(lookback = 22, short = 10, long = 22)
    temp = TA.Factor_MADMOM(lookback = 22, short = 3, long = 200)
    temp = TA.Factor_MADMOM(lookback = 22, short = 10, long = 200)
    temp = TA.Factor_MADMOM(lookback = 22, short = 22, long = 200)
    temp = TA.Factor_MAD_std(short = 3, long = 22)
    temp = TA.Factor_MAD_std(short = 3, long = 66)
    temp = TA.Factor_MAD_std(short = 5, long = 22)
    temp = TA.Factor_MAD_std(short = 5, long = 66)
    temp = TA.Factor_MAD_std(short = 10, long = 22)
    temp = TA.Factor_MAD_std(short = 10, long = 66)
    temp = TA.Factor_MAD_std(short = 3, long = 200)
    temp = TA.Factor_MAD_std(short = 10, long = 200)
    temp = TA.Factor_MAD_std(short = 22, long = 200)
    temp = TA.Factor_MOM(window = 3)
    temp = TA.Factor_MOM(window = 5)
    temp = TA.Factor_MOM(window = 10)
    temp = TA.Factor_MOM(window = 22)
    temp = TA.Factor_MOM(window = 66)
    temp = TA.Factor_MOM(window = 132)
    temp = TA.Factor_OBV(short = 3, long = 10, rolling = 5)
    temp = TA.Factor_OBV(short = 3, long = 10, rolling = 10)
    temp = TA.Factor_OBV(short = 3, long = 10, rolling = 22)
    temp = TA.Factor_OBV(short = 5, long = 22, rolling = 5)
    temp = TA.Factor_OBV(short = 5, long = 22, rolling = 10)
    temp = TA.Factor_OBV(short = 5, long = 22, rolling = 22)
    temp = TA.Factor_OBV(short = 10, long = 22, rolling = 5)
    temp = TA.Factor_OBV(short = 10, long = 22, rolling = 10)
    temp = TA.Factor_OBV(short = 10, long = 22, rolling = 22)
    temp = TA.Factor_OBV2(short = 3, long = 10, rolling = 5)
    temp = TA.Factor_OBV2(short = 3, long = 10, rolling = 10)
    temp = TA.Factor_OBV2(short = 3, long = 10, rolling = 22)
    temp = TA.Factor_OBV2(short = 5, long = 22, rolling = 5)
    temp = TA.Factor_OBV2(short = 5, long = 22, rolling = 10)
    temp = TA.Factor_OBV2(short = 5, long = 22, rolling = 22)
    temp = TA.Factor_OBV2(short = 10, long = 22, rolling = 5)
    temp = TA.Factor_OBV2(short = 10, long = 22, rolling = 10)
    temp = TA.Factor_OBV2(short = 10, long = 22, rolling = 22)
    temp = TA.Factor_PPSR(lookback = 22)
    temp = TA.Factor_PPSR(lookback = 66)
    temp = TA.Factor_PPSR(lookback = 132)
    temp = TA.Factor_RSI_bot(short = 3, long = 10, rolling = 5)
    temp = TA.Factor_RSI_bot(short = 3, long = 10, rolling = 10)
    temp = TA.Factor_RSI_bot(short = 3, long = 10, rolling = 22)
    temp = TA.Factor_RSI_bot(short = 5, long = 22, rolling = 5)
    temp = TA.Factor_RSI_bot(short = 5, long = 22, rolling = 10)
    temp = TA.Factor_RSI_bot(short = 5, long = 22, rolling = 22)
    temp = TA.Factor_RSI_bot(short = 10, long = 22, rolling = 5)
    temp = TA.Factor_RSI_bot(short = 10, long = 22, rolling = 10)
    temp = TA.Factor_RSI_bot(short = 10, long = 22, rolling = 22)
    temp = TA.Factor_RSI_low(window = 5)
    temp = TA.Factor_RSI_low(window = 10)
    temp = TA.Factor_RSI_low(window = 22)
    temp = TA.Factor_Reversal(window = 3)
    temp = TA.Factor_Reversal(window = 5)
    temp = TA.Factor_Reversal(window = 10)
    temp = TA.Factor_Reversal(window = 22)
    temp = TA.Factor_Reversal(window = 66)
    temp = TA.Factor_Reversal(window = 132)
    temp = TA.Factor_STO(window = 5, rolling = 5)
    temp = TA.Factor_STO(window = 14, rolling = 5)
    temp = TA.Factor_STO(window = 22, rolling = 5)
    temp = TA.Factor_STO(window = 5, rolling = 10)
    temp = TA.Factor_STO(window = 14, rolling = 10)
    temp = TA.Factor_STO(window = 22, rolling = 10)
    temp = TA.Factor_TRIX(short = 3, long = 10, rolling = 5)
    temp = TA.Factor_TRIX(short = 3, long = 10, rolling = 10)
    temp = TA.Factor_TRIX(short = 3, long = 10, rolling = 22)
    temp = TA.Factor_TRIX(short = 5, long = 22, rolling = 5)
    temp = TA.Factor_TRIX(short = 5, long = 22, rolling = 10)
    temp = TA.Factor_TRIX(short = 5, long = 22, rolling = 22)
    temp = TA.Factor_TRIX(short = 10, long = 22, rolling = 5)
    temp = TA.Factor_TRIX(short = 10, long = 22, rolling = 10)
    temp = TA.Factor_TRIX(short = 10, long = 22, rolling = 22)
    temp = TA.Factor_TRIX_ret(short = 3, long = 10, rolling = 5)
    temp = TA.Factor_TRIX_ret(short = 3, long = 10, rolling = 10)
    temp = TA.Factor_TRIX_ret(short = 3, long = 10, rolling = 22)
    temp = TA.Factor_TRIX_ret(short = 5, long = 22, rolling = 5)
    temp = TA.Factor_TRIX_ret(short = 5, long = 22, rolling = 10)
    temp = TA.Factor_TRIX_ret(short = 5, long = 22, rolling = 22)
    temp = TA.Factor_TRIX_ret(short = 10, long = 22, rolling = 5)
    temp = TA.Factor_TRIX_ret(short = 10, long = 22, rolling = 10)
    temp = TA.Factor_TRIX_ret(short = 10, long = 22, rolling = 22)
    temp = TA.Factor_TRIX_vol(short = 3, long = 10, rolling = 5)
    temp = TA.Factor_TRIX_vol(short = 3, long = 10, rolling = 10)
    temp = TA.Factor_TRIX_vol(short = 3, long = 10, rolling = 22)
    temp = TA.Factor_TRIX_vol(short = 5, long = 22, rolling = 5)
    temp = TA.Factor_TRIX_vol(short = 5, long = 22, rolling = 10)
    temp = TA.Factor_TRIX_vol(short = 5, long = 22, rolling = 22)
    temp = TA.Factor_TRIX_vol(short = 10, long = 22, rolling = 5)
    temp = TA.Factor_TRIX_vol(short = 10, long = 22, rolling = 10)
    temp = TA.Factor_TRIX_vol(short = 10, long = 22, rolling = 22)
    
    # backtest
    CSI500 = pd.Series(io.loadmat(f'{data_dir}csi500_close.mat').popitem()[1].reshape([-1]))
    CSI500_open = pd.Series(io.loadmat(f'{data_dir}csi500_open.mat').popitem()[1].reshape([-1]))
    Close = pd.DataFrame(io.loadmat(f'{data_dir}data_close_price.mat').popitem()[1])
    VWAP = pd.DataFrame(io.loadmat(f'{data_dir}data_vwap_price.mat').popitem()[1])
    Check = io.loadmat(f'{data_dir}data_tickercheck.mat').popitem()[1]
    MV = io.loadmat(f'{data_dir}data_mv.mat').popitem()[1]
    
    Close[Close == 0] = np.nan
    VWAP[VWAP == 0] = np.nan
    CSI500[CSI500 == 0] = np.nan
    CSI500_open[CSI500_open == 0] = np.nan
    Close_fill = Close.ffill(axis=1)
    High = io.loadmat(f'{data_dir}data_high_price.mat').popitem()[1]
    Low = io.loadmat(f'{data_dir}data_low_price.mat').popitem()[1]
    cant = High == Low
    cant_buy = cant * (Low / Close_fill.shift(axis=1) - 1 > 0.09).values
    cant_sell = cant * (High / Close_fill.shift(axis=1) - 1 < -0.09).values
    Close_fill = Close_fill.values
    Close = Close.values
    VWAP = VWAP.values
    CSI500 = CSI500.ffill().bfill()
    CSI500_open = CSI500_open.ffill().bfill()
    MV_weight = MV / np.sum(MV, axis=0)
    
    fturn = io.loadmat(f'{data_dir}data_fturn.mat').popitem()[1] / 100
    fmv = io.loadmat(f'{data_dir}data_fmv.mat').popitem()[1]
    fliq_top = pd.DataFrame(fturn * fmv).rolling(5,axis=1).mean()
    fliq_top[pd.isnull(fliq_top)] = pd.DataFrame(fturn * fmv)[pd.isnull(fliq_top)]
    fliq_top = fliq_top.values

BBands_factor = TA.Factor_BBands()
TRIX_vol_factor = TA.Factor_TRIX_vol()
TRIX_ret_factor = TA.Factor_TRIX_ret()
TRIX_factor = TA.Factor_TRIX()
MACD_factor = TA.Factor_MACD()
RSI_low_factor = TA.Factor_RSI_low()
RSI_bot_factor = TA.Factor_RSI_bot()
after_factor = TA.ret_afternoon()

gg = pd.Series(aum) / pd.Series(aum).shift() - 1
np.array(Tickers)[np.where(weights[:,358] > 0)[0]]
p = (pd.DataFrame(Close) / pd.DataFrame(Close).shift(axis=1)).loc[weights[:,616] > 0, 617] - (CSI500 / CSI500.shift())[617]
(p>0).sum()/len(p)
p.mean()
gg[616]

d_h = (holding[:,358] - holding[:,357])
rel_ret = ((pd.DataFrame(Close) / pd.DataFrame(VWAP)).iloc[:, 358] - (CSI500 / CSI500.shift())[358])
p1 = rel_ret[reb_value>0].mean()
p2 = rel_ret[reb_value<0].mean()
np.nansum(rel_ret*d_h) / np.nansum(np.abs(d_h))

 * (pd.DataFrame(Close) / pd.DataFrame(Close).shift(axis=1)).loc[:, 617] - (CSI500 / CSI500.shift())[617]

siw = pd.read_csv('D:/Projects/ANN_3y/results/score_1.csv', index_col=0, header=0)
date_map = [Dates.index(str(x)) for x in siw.index]
ticker_map = [Tickers.index(x) for x in siw.columns]
Dates = np.array(Dates)[date_map].tolist()
Tickers = np.array(Tickers)[ticker_map].tolist()
Close = Close[ticker_map,:][:,date_map]
CSI500 = CSI500[date_map].reset_index(drop=True)
CSI500_open = CSI500_open[date_map].reset_index(drop=True)
Close_fill = Close_fill[ticker_map,:][:,date_map]
VWAP = VWAP[ticker_map,:][:,date_map]
High = High[ticker_map,:][:,date_map]
Low = Low[ticker_map,:][:,date_map]
Check = Check[ticker_map,:][:,date_map]
cant_buy = cant_buy[ticker_map,:][:,date_map]
cant_sell = cant_sell[ticker_map,:][:,date_map]
fliq_top = fliq_top[ticker_map,:][:,date_map]
Check = Check[ticker_map,:][:,date_map]
siw = siw.T.values

 #%%
    siw[Check==0] = -10000
    weights = np.zeros(siw.shape)
    for i in range(siw.shape[1]):
        if i < 1:
            continue
        top = np.argpartition(siw[:,i-1], -100)[-100:]
        weights[top,i] = 1#MV[top,i-1]
    weights /= np.sum(weights,axis=0)
    weights[np.isnan(weights)] = 0

    bm = CSI500 / CSI500.shift(1)
    cash_total = 100000000

    year_point = np.where(pd.Series([int(x[:4]) for x in Dates]).diff() == 1)[0]

    occ = []
    hocc = []
    holding = np.zeros(Close.shape)
    future_holding = np.zeros(Close.shape[1])
    cash_lend = 0
    topV = cash_total
    DD = np.zeros(Close.shape[1])
    maxDD = 0
    risk_level = np.zeros(Close.shape[1])
    cash = np.zeros(Close.shape[1])
    cash[0] = cash_total
    aum = np.zeros(Close.shape[1])
    aum[0] = cash_total
    model_line = aum.copy()
    NV_today = np.zeros(Close.shape[0])
    pb = ProgressBar()
    for i in pb(range(Close.shape[1])):
    
        if i < 1:
            occ.append(0)
            hocc.append(0)
            continue

        cash[i] = cash[i-1]
        holding[:,i] = holding[:,i-1]
#        if i == 2318:
#            break
        # lend
        if cash[i] < 0:
            cash_lend -= cash[i]
            cash[i] = 0
        
        # repay
        if cash[i] > 0 and cash_lend > 0:
            if cash[i] < cash_lend:
                cash_lend -= cash[i]
                cash[i] = 0
            else:
                cash[i] -= cash_lend
                cash_lend = 0
        
        if risk_level[i-1] == 0:
            risk_level[i] = 1*(DD[i-1] <= -0.02) + 1*(DD[i-1] <= -0.04)
        else:
            risk_level[i] = risk_level[i-1] - (risk_level[i-1]-1)*(DD[i-1] >= -0.03) - 1*(DD[i-1] >= -0.01) + (2-risk_level[i-1])*(DD[i-1] <= -0.04)
        
        weight_v = weights[:,i]
        liq_mask = fliq_top[:,i]/(0.8*aum[i-1])
        liq_mask[~(liq_mask < 0.05)] = 0.05
        liq_mask[liq_mask == 0] = 0.05
        balance_mask = weight_v > liq_mask
        if balance_mask.any():
            remain = (weight_v[balance_mask] - liq_mask[balance_mask]).sum()
            weight_v[balance_mask] = liq_mask[balance_mask]
            weight_v[~balance_mask] *= (remain / weight_v[~balance_mask].sum() + 1)
        balance_mask = weight_v > liq_mask
        if balance_mask.any():
            remain = (weight_v[balance_mask] - liq_mask[balance_mask]).sum()
            weight_v[balance_mask] = liq_mask[balance_mask]
            weight_v[~balance_mask] *= (remain / weight_v[~balance_mask].sum() + 1)
        weight_v /= weight_v.sum()
        
        h_yes = holding[:,i-1] * Close_fill[:,i-1]
        h_yes[np.isnan(Close_fill[:,i-1])] = 0
        build_speed = 0.8
        reb_value = build_speed*aum[i-1]*weight_v - h_yes
        reb_value[reb_value < 0] *= ~cant_sell[reb_value < 0,i]
        reb_value[reb_value > 0] *= ~cant_buy[reb_value > 0,i]

        liq_mask = np.abs(reb_value) >= fliq_top[:,i-1] * 0.1
        if np.sum(liq_mask) > 0:
            reb_value[liq_mask] = np.sign(reb_value[liq_mask])*fliq_top[liq_mask,i-1]*0.1
        
        turn_mask = np.abs(reb_value / h_yes) < 0.1
        reb_value[turn_mask] = 0
        
        future_holding[i] = np.round((reb_value+h_yes).sum() / CSI500[i-1] / 200)
        
        reb_buy_mask = (reb_value > 0) * (h_yes > 0)
        reb_sell_mask = (reb_value < 0) * (reb_value + h_yes > 0)
        all_buy_mask = (reb_value > 0) * (h_yes == 0)
        all_sell_mask = (reb_value < 0) * (reb_value + h_yes == 0)
        
#        cash_line = 0.15 * aum[i-1]
        
        trn = []
        
        # rebalance sell
        if np.sum(reb_sell_mask) > 0:
            d_rb_s = np.round(reb_value[reb_sell_mask] / Close[reb_sell_mask,i-1],-2)
            d_rb_s[np.isnan(d_rb_s)] = 0
            holding[reb_sell_mask,i] += d_rb_s
            cash[i] += np.nansum(-d_rb_s * VWAP[reb_sell_mask,i]) * (1-0.0002-0.001) # 72984458            

        # repay
        if cash[i] > 0 and cash_lend > 0:
            if cash[i] < cash_lend:
                cash_lend -= cash[i]
                cash[i] = 0
            else:
                cash[i] -= cash_lend
                cash_lend = 0
        
        # rebalance buy
        if np.sum(reb_buy_mask) > 0 and cash[i] > 0:
            reb_value[reb_buy_mask] /= ((np.sum(reb_value[reb_buy_mask]) / (cash[i])) if np.sum(reb_value[reb_buy_mask]) > (cash[i]) else 1)
            d_rb_b = np.round(reb_value[reb_buy_mask] / Close[reb_buy_mask,i-1],-2)
            d_rb_b[np.isnan(d_rb_b)] = 0
            holding[reb_buy_mask,i] += d_rb_b
            cash[i] -= np.nansum(d_rb_b * VWAP[reb_buy_mask,i]) * (1+0.0002)
            
        # lend
        if cash[i] < 0:
            cash_lend -= cash[i]
            cash[i] = 0
        
        # sell
        if np.sum(all_sell_mask) > 0:
            cash[i] += np.nansum(holding[all_sell_mask,i] * VWAP[all_sell_mask,i]) * (1-0.0002-0.001) # 111391565
            holding[all_sell_mask,i] = 0
            

        # repay
        if cash[i] > 0 and cash_lend > 0:
            if cash[i] < cash_lend:
                cash_lend -= cash[i]
                cash[i] = 0
            else:
                cash[i] -= cash_lend
                cash_lend = 0
        
        # buy
        if np.sum(all_buy_mask) > 0 and cash[i] > 0:
            reb_value[all_buy_mask] /= ((np.sum(reb_value[all_buy_mask]) / (cash[i])) if np.sum(reb_value[all_buy_mask]) > (cash[i]) else 1)
            d_all_b = np.floor(reb_value[all_buy_mask] / VWAP[all_buy_mask,i-1]/100)*100
            d_all_b[np.isnan(d_all_b)] = 0
            holding[all_buy_mask,i] = d_all_b
            cash[i] -= np.nansum(d_all_b * VWAP[all_buy_mask,i]) * (1+0.0002) # 61828839
        
    
        d_f = future_holding[i] - future_holding[i-1]
        cash[i] += (200*d_f*((CSI500_open[i] + CSI500[i])/2 - CSI500[i]))
        cash[i] += (200*future_holding[i-1]*(CSI500[i-1] - CSI500[i]))
        
        holding_v = holding[:,i] * Close_fill[:,i]
        holding_v[np.isnan(Close_fill[:,i])] = 0
        cash[i] -= aum[i-1]* 0.0001
        aum[i] = (holding_v.sum() + cash[i] - cash_lend) 
        occ.append(200*d_f*((CSI500_open[i] + CSI500[i])/2 - CSI500[i])+200*future_holding[i-1]*(CSI500[i-1] - CSI500[i]))
        hocc.append(holding_v.sum() / aum[i])
        if i in year_point:
            topV = 0
            maxDD = 0
        if aum[i] > topV:
            topV = aum[i]
        else:
            DD[i] = aum[i] / topV - 1
            if DD[i] < maxDD:
                maxDD = DD[i]
    
    temp_DD = DD.copy()
    temp_aum = aum.copy()
    temp_hocc = np.array(hocc)

    occ = []
    hocc = []
    holding = np.zeros(Close.shape)
    future_holding = np.zeros(Close.shape[1])
    cash_lend = 0
    topV = cash_total
    DD = np.zeros(Close.shape[1])
    maxDD2 = 0
    cash = np.zeros(Close.shape[1])
    cash[0] = cash_total
    aum = np.zeros(Close.shape[1])
    aum[0] = cash_total
    NV_today = np.zeros(Close.shape[0])
    pb = ProgressBar()
    for i in pb(range(Close.shape[1])):
        if i < 1:
            occ.append(0)
            hocc.append(0)
            continue

        cash[i] = cash[i-1]
        holding[:,i] = holding[:,i-1]
#        if i == 1688:
#            break
        # lend
        if cash[i] < 0:
            cash_lend -= cash[i]
            cash[i] = 0
        
        # repay
        if cash[i] > 0 and cash_lend > 0:
            if cash[i] < cash_lend:
                cash_lend -= cash[i]
                cash[i] = 0
            else:
                cash[i] -= cash_lend
                cash_lend = 0
        
        weight_v = weights[:,i]
        liq_mask = fliq_top[:,i]/(0.8*aum[i-1]*(1-0.5*risk_level[i]))
        liq_mask[~(liq_mask < 0.05)] = 0.05
        liq_mask[liq_mask == 0] = 0.05
        balance_mask = weight_v > liq_mask
        if balance_mask.any():
            remain = (weight_v[balance_mask] - liq_mask[balance_mask]).sum()
            weight_v[balance_mask] = liq_mask[balance_mask]
            weight_v[~balance_mask] *= (remain / weight_v[~balance_mask].sum() + 1)
        balance_mask = weight_v > liq_mask
        if balance_mask.any():
            remain = (weight_v[balance_mask] - liq_mask[balance_mask]).sum()
            weight_v[balance_mask] = liq_mask[balance_mask]
            weight_v[~balance_mask] *= (remain / weight_v[~balance_mask].sum() + 1)
        weight_v /= weight_v.sum()
        
        h_yes = holding[:,i-1] * Close_fill[:,i-1]
        h_yes[np.isnan(Close_fill[:,i-1])] = 0
        build_speed = 0.8
        reb_value = build_speed*aum[i-1]*weight_v*(1-0.5*risk_level[i]) - h_yes
        reb_value[reb_value < 0] *= ~cant_sell[reb_value < 0,i]
        reb_value[reb_value > 0] *= ~cant_buy[reb_value > 0,i]

        liq_mask = np.abs(reb_value) >= fliq_top[:,i-1] * 0.1
        if np.sum(liq_mask) > 0:
            reb_value[liq_mask] = np.sign(reb_value[liq_mask])*fliq_top[liq_mask,i-1]*0.1
        
        turn_mask = np.abs(reb_value / h_yes) < 0.1
        reb_value[turn_mask] = 0
        
        future_holding[i] = np.round((reb_value+h_yes).sum() / CSI500[i-1] / 200)
        
        reb_buy_mask = (reb_value > 0) * (h_yes > 0)
        reb_sell_mask = (reb_value < 0) * (reb_value + h_yes > 0)
        all_buy_mask = (reb_value > 0) * (h_yes == 0)
        all_sell_mask = (reb_value < 0) * (reb_value + h_yes == 0)
        
#        cash_line = 0.15 * aum[i-1]
        
        trn = []
        
        # rebalance sell
        if np.sum(reb_sell_mask) > 0:
            d_rb_s = np.round(reb_value[reb_sell_mask] / Close[reb_sell_mask,i-1],-2)
            d_rb_s[np.isnan(d_rb_s)] = 0
            holding[reb_sell_mask,i] += d_rb_s
            cash[i] += np.nansum(-d_rb_s * VWAP[reb_sell_mask,i]) * (1-0.0002-0.001) # 72984458
            

        # repay
        if cash[i] > 0 and cash_lend > 0:
            if cash[i] < cash_lend:
                cash_lend -= cash[i]
                cash[i] = 0
            else:
                cash[i] -= cash_lend
                cash_lend = 0
        
        # rebalance buy
        if np.sum(reb_buy_mask) > 0 and cash[i] > 0:
            reb_value[reb_buy_mask] /= ((np.sum(reb_value[reb_buy_mask]) / (cash[i])) if np.sum(reb_value[reb_buy_mask]) > (cash[i]) else 1)
            d_rb_b = np.round(reb_value[reb_buy_mask] / Close[reb_buy_mask,i-1],-2)
            d_rb_b[np.isnan(d_rb_b)] = 0
            holding[reb_buy_mask,i] += d_rb_b
            cash[i] -= np.nansum(d_rb_b * VWAP[reb_buy_mask,i]) * (1+0.0002)
            
        # lend
        if cash[i] < 0:
            cash_lend -= cash[i]
            cash[i] = 0
        
        # sell
        if np.sum(all_sell_mask) > 0:
            trn.append(np.nansum(holding[all_sell_mask,i] * VWAP[all_sell_mask,i]) * (1-0.0002-0.001)-np.sum(holding[all_sell_mask,i] * Close[all_sell_mask,i]))
            cash[i] += np.nansum(holding[all_sell_mask,i] * VWAP[all_sell_mask,i]) * (1-0.0002-0.001) # 111391565
            holding[all_sell_mask,i] = 0
            

        # repay
        if cash[i] > 0 and cash_lend > 0:
            if cash[i] < cash_lend:
                cash_lend -= cash[i]
                cash[i] = 0
            else:
                cash[i] -= cash_lend
                cash_lend = 0
        
        # buy
        if np.sum(all_buy_mask) > 0 and cash[i] > 0:
            reb_value[all_buy_mask] /= ((np.sum(reb_value[all_buy_mask]) / (cash[i])) if np.sum(reb_value[all_buy_mask]) > (cash[i]) else 1)
            d_all_b = np.floor(reb_value[all_buy_mask] / VWAP[all_buy_mask,i-1]/100)*100
            d_all_b[np.isnan(d_all_b)] = 0
            holding[all_buy_mask,i] = d_all_b
            cash[i] -= np.nansum(d_all_b * VWAP[all_buy_mask,i]) * (1+0.0002) # 61828839
            trn.append(np.sum(d_all_b * Close[all_buy_mask,i])-np.nansum(d_all_b * VWAP[all_buy_mask,i]) * (1+0.0002))
        
    
        d_f = future_holding[i] - future_holding[i-1]
        cash[i] += (200*d_f*((CSI500_open[i] + CSI500[i])/2 - CSI500[i]))
        cash[i] += (200*future_holding[i-1]*(CSI500[i-1] - CSI500[i]))
        
        holding_v = holding[:,i] * Close_fill[:,i]
        holding_v[np.isnan(Close_fill[:,i])] = 0
        cash[i] -= aum[i-1]* 0.0001
        aum[i] = (holding_v.sum() + cash[i] - cash_lend) 
        occ.append(200*d_f*((CSI500_open[i] + CSI500[i])/2 - CSI500[i])+200*future_holding[i-1]*(CSI500[i-1] - CSI500[i]))
        hocc.append(holding_v.sum() / aum[i])
        if i in year_point:
            topV = 0
            maxDD2 = 0
        if aum[i] > topV:
            topV = aum[i]
        else:
            DD[i] = aum[i] / topV - 1
            if DD[i] < maxDD2:
                maxDD2 = DD[i]
        
    
    dps = pd.Series(aum,index=pd.to_datetime(Dates))
    temp_dps = pd.Series(temp_aum,index=pd.to_datetime(Dates))
    ax = (dps / dps[0]).plot()
    (temp_dps / temp_dps[0]).plot()
    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                  risk_level[np.newaxis], cmap='Reds', alpha=0.2)
#    (1-pd.Series(np.array(occ) / aum / 0.8)).cumprod().plot()
#bbb = np.diff(aum) - np.diff(aum_temp)
#ppp = (holding[:,2885] - holding[:,2884]) - (holding_temp[:,2885] - holding_temp[:,2884])
#g = pd.DataFrame(Close)
#g = g / g.shift(axis=1) - 1
#buy = pd.DataFrame(Close) / pd.DataFrame(VWAP) - 1
#sell = pd.DataFrame(VWAP) / pd.DataFrame(Close).shift(axis=1) - 1
#
#np.sum(pd.DataFrame(Close).shift(axis=1).iloc[np.where(ppp > 0)[0],2885]*np.sum(sell.iloc[np.where(ppp > 0)[0],2885] * ppp[np.where(ppp > 0)[0]]) )#/ ppp[ppp>0].sum()
#np.sum(pd.DataFrame(VWAP).iloc[np.where(ppp < 0)[0],2885]*np.sum(buy.iloc[np.where(ppp < 0)[0],2885] * -ppp[np.where(ppp < 0)[0]]))# / -ppp[ppp<0].sum()
#aum = pd.Series(aum, index = pd.to_datetime(Dates))
#aum_temp = pd.Series(aum_temp, index = pd.to_datetime(Dates))
#
#year = '2008'
#aum[year][-1] / aum[year][0] - aum_temp[year][-1] / aum_temp[year][0]

#aum_temp = aum.copy()
#holding_temp = holding.copy()

#%%       
        
    gg = (holding[:,1999] > 0).sum(axis=0)
        
        
        holding[:,1999]
    
    
        tttt = ttt[np.array(Tickers)[holding[:,2297] > 0]]
        -np.sort(-siw[:,1999])[:100]
        ttt = pd.Series(siw[:,1999],index=Tickers)
        fs = aum[1:] / aum[:-1] - 1
        threshold = np.nanpercentile(siw[:,i-1], 90)
        if np.isnan(threshold):
            continue
        signal = (siw[:,i-1] >= threshold)*1
        signal[np.isnan(signal)] = 0
        signal[Check[:,i-1]==0] = holding[Check[:,i-1]==0]
        pnl[:,i] += (1-holding)*signal * (Close[:,i]/VWAP[:,i]-1)
        pnl[:,i] += holding*signal * (Close[:,i]/Close[:,i-1]-1)
        pnl[:,i] += holding*(1-signal) * (VWAP[:,i]/Close[:,i-1]-1)
        pnl_flag = (signal==1) + (holding==1)
        temp_weight = MV_weight[pnl_flag,i] / np.sum(MV_weight[pnl_flag,i])
        pnl[pnl_flag,i] *= temp_weight
        if (pnl[:,i]>1).any():
            raise TypeError('1')
#        temp_holding = holding[Check[:,i-1]==0].copy()
        trn[i] = np.nansum(holding != signal) / np.nansum(holding)
        holding = signal.copy()
#        holding[Check[:,i-1]==0] = temp_holding
    trn = np.array(trn)
    trn[np.isinf(trn)] = 0
    pnl[np.isnan(pnl)] = 0
    pd.Series(np.sum(pnl,axis=0)+1-trn*1.4/1000).cumprod().plot()
    pd.Series(bm).cumprod().plot()
    pd.Series(np.nansum(all_pnl,axis=0)+1).cumprod().plot()
    
    pd.Series(np.sum(pnl, axis=0)-bm+2-trn*1.4/1000).cumprod().plot()
    pd.Series(np.nansum(all_pnl, axis=0)-bm+2).cumprod().plot()
    
    pd.Series(pnl[232,:]+1).cumprod().plot()
    ExcRet = (Close / VWAP.shift(1,axis=1) - CSI500 / CSI500.shift(1)).shift(2,axis=1)
    ExcRet[np.isinf(ExcRet)] = np.nan
    
    ExcRet *= Check.shift(1,axis=1)
    
    qq = (g >= np.nanpercentile(g, q=90, axis=0).reshape([1,-1])) * ExcRet
    (1+qq.mean(axis=0)).cumprod().plot()
    
    table = pd.DataFrame(Data[:,1,:].T, columns=data_col, index=Dates)
    table['trading_value'] = table['turn'] * table['mv'] / 100
    table['volume'] = table['trading_value'] / table['close_price']
    table[table['turn'] == 0] = np.nan
    
    
    
    colname = table.columns.tolist()
    rowname = table.index.tolist()
    
    series = table['close_price'][:100]

    SMA(table, 60)
    EMA(table, 50, 0.9)
    MOM(table, 12)
    ROC(table, 12)
    ATR(table, 5)