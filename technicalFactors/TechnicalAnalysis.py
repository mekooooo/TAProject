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
        h5.create_dataset(key, data=np.array(np.array(data,
                                                      dtype=np.dtype(str)), 
    dtype=h5py.special_dtype(vlen=str)), dtype=h5py.special_dtype(vlen=str))
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
        
        if 'Amount' not in Data:
            Data['Amount'] = Data['Turnover']*Data['MV']/Data['Close']/100
        
        na_mask = ~(Data['Amount'] != 0)
        Data['Amount'][na_mask] = np.nan
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
        OBV = temp * self.Data['Volume']
        OBV = OBV.rolling(window, axis=0).mean()
        return OBV
    
    @staticmethod
    def TSCov(df1, df2, n):
        EX = df1.rolling(n, axis=0).mean()
        EY = df2.rolling(n, axis=0).mean()
        EXY = (df1*df2).rolling(n, axis=0).mean()
        return EXY - EX*EY
    
    @staticmethod
    def TSCorr(df1, df2, n):
        EX = df1.rolling(n, axis=0).mean()
        EY = df2.rolling(n, axis=0).mean()
        EXY = (df1*df2).rolling(n, axis=0).mean()
        varX = df1.rolling(n, axis=0).var()
        varY = df2.rolling(n, axis=0).var()
        return (EXY - EX*EY)/np.sqrt(varX*varY)*n/(n-1)
    
    @staticmethod
    def TSRank(df, n):
        return df.rolling(n, axis=0).apply(lambda x:
            1 - x.argsort().argsort()[-1] / x.size)
    
    @staticmethod
    def DecayLinear(df):
        weights = np.arange(len(df))+1
        return (df * weights).sum() / weights.sum()
    
    @staticmethod
    def HighDay(df):
        return len(df) - df.argmax()
    
    @staticmethod
    def LowDay(df):
        return len(df) - df.argmin()
    
    @staticmethod
    def RegBeta(df1, df2):
        n = df2.shape[0]
        if n < df1.shape[0]:
            def Beta(d1, d2):
                XY = d1 * d2
                XX = d1 ** 2
                return ((n*XY.sum() -
                        d2.sum() * d1.sum())/
                (n*XX.sum() - (d1.sum())**2))
            return df1.rolling(n).apply(lambda x: Beta(x, df2))
        else:
            XY = df1.mul(df2, axis=0)
            XX = df1 ** 2
            return (n*XY.rolling(n, axis=0).sum() -
                    df1.rolling(n, axis=0).sum().mul(df2.rolling(n, axis=0).sum(),
                              axis=0)).div(n*XX.rolling(n, axis=0).sum()
                    - (df1.rolling(n, axis=0).sum())**2, axis=0)
    
    @staticmethod
    def RegResid(df1, df2):
        n = df2.shape[0]
        if n < df1.shape[0]:
            def Beta(d1, d2):
                XY = d1 * d2
                XX = d1 ** 2
                return ((n*XY.sum() -
                        d2.sum() * d1.sum())/
                (n*XX.sum() - (d1.sum())**2))
            return df1.rolling(n).apply(lambda x: Beta(x, df2))
        else:
            XY = df2.mul(df1, axis=0)
            XX = df1 ** 2
            beta = (n*XY.rolling(n, axis=0).sum() -
                    df2.rolling(n, axis=0).sum().mul(df1.rolling(n, axis=0).sum(),
                              axis=0)).div(n*XX.rolling(n, axis=0).sum()
                    - (df1.rolling(n, axis=0).sum())**2, axis=0) 
            alpha = (df2.rolling(n, axis=0).mean() -
                     beta * df1.rolling(n, axis=0).mean())
            return df2 - beta.mul(df1, axis=0) - alpha
        

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
                'Amount': from_hdf_df(
                        '{}/Amount.h5'.format(h5_dir), 'Raw'),
                'Volume': from_hdf_df(
                        '{}/Volume.h5'.format(h5_dir), 'Raw'),
                'Index': pd.Series(from_hdf(
                        '{}/CSI500.h5'.format(h5_dir), 'Close'),
                        index=pd.to_datetime(from_hdf(
                                '{}/CSI500.h5'.format(h5_dir), 'Dates'))
                        ).reindex(pd.to_datetime(Dates)),
                'IndexOpen': pd.Series(from_hdf(
                        '{}/CSI500.h5'.format(h5_dir), 'Open'),
                        index=pd.to_datetime(from_hdf(
                                '{}/CSI500.h5'.format(h5_dir), 'Dates'))
                        ).reindex(pd.to_datetime(Dates))
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
                if res is None:
                    return res
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
        for i in range(close.shape[1]):
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
    
    @Output()
    def Factor_GTJA_1(self):
        return -self.TSCorr(np.log(self.Data['Volume']).diff(axis=0).rank(
                axis=1, pct=True), (self.Data['Close']/self.Data['Open'] -
                                1).rank(axis=1, pct=True), 6)
    
    @Output()
    def Factor_GTJA_2(self):
        return -((self.Data['Close']*2 - self.Data['Low'] - self.Data['High'])/
                (self.Data['High'] - self.Data['Low'])).diff(axis=0)
    
    @Output()
    def Factor_GTJA_3(self):
        delay = self.Data['Close'].shift(axis=0)
        part1 = self.Data['Close']*(self.Data['Close'].diff(axis=0) != 0)
        temp1 = np.minimum(self.Data['Low'], delay)
        temp2 = np.maximum(self.Data['High'], delay)
        part2 = temp1.where(self.Data['Close'] > delay, temp2)
        return (part1 - part2).rolling(6, axis=0).sum()
    
    @Output()
    def Factor_GTJA_4(self):
        c_mean8 = self.Data['Close'].rolling(8, axis=0).mean()
        c_mean2 = self.Data['Close'].rolling(2, axis=0).mean()
        c_std8 = self.Data['Close'].rolling(8, axis=0).std()
        v_mean20 = self.Data['Volume'].rolling(20, axis=0).mean()
        temp = pd.DataFrame(-1, index=c_mean8.index, columns=c_mean8.columns)
        temp[(~((c_mean8 + c_std8) < c_mean2) &
              ((c_mean2 < (c_mean8 - c_std8)) |
                      ((self.Data['Volume'] / v_mean20) >= 1)))] = 1
        return temp
    
    @Output()
    def Factor_GTJA_5(self):
        return -(self.TSCorr(self.TSRank(self.Data['Volume'], 5),
                             self.TSRank(self.Data['High'], 3),
                             5)).rolling(3, axis=0).max()     
        
    @Output()
    def Factor_GTJA_6(self):
        return -np.sign((self.Data['Open']*0.85 +
                         self.Data['High']*0.15).diff(4, 
                                  axis=0)).rank(axis=1, pct=True)
    
    @Output()
    def Factor_GTJA_7(self):
        return ((self.Data['VWAP'] - self.Data['Close']).rolling(3, 
                axis=0).max().rank(axis=1, pct=True) + ((self.Data['VWAP'] -
                           self.Data['Close']).rolling(3, axis=0).min().rank(
                                   axis=1, pct=True)) * 
            (self.Data['Volume'].diff(3, axis=0).rank(axis=1, pct=True)))
    
    @Output()
    def Factor_GTJA_8(self):
        return -((self.Data['High'] + self.Data['Low'])*0.1 +
                 self.Data['VWAP']*0.8).diff(4, axis=0).rank(axis=1, pct=True)
    
    @Output()
    def Factor_GTJA_9(self):
        return (((self.Data['High']+self.Data['Low'])/2 -
                 (self.Data['High'].shift(axis=0) +
                  self.Data['Low'].shift(axis=0))/2) * 
            (self.Data['High']-self.Data['Low']) / 
            self.Data['Volume']).ewm(alpha=2/7, axis=0).mean()
    
    @Output()
    def Factor_GTJA_10(self):
        ret = self.Data['Close'].diff(axis=0)
        return (ret.rolling(20, axis=0).std().where(ret<0,
                self.Data['Close'])**2).rolling(5, 
            axis=0).max().rank(axis=1, pct=True)
        
    @Output()
    def Factor_GTJA_12(self):
        return (self.Data['Open'] - self.Data['VWAP'].rolling(10,
                axis=0).mean()).rank(axis=1, pct=True)*(-(self.Data['Close'] -
                            self.Data['VWAP']).abs().rank(axis=1, pct=True))
    
    @Output()
    def Factor_GTJA_13(self):
        return np.sqrt(self.Data['High']*self.Data['Low']) - self.Data['VWAP']
    
    @Output()
    def Factor_GTJA_14(self):
        return self.Data['Close'].diff(5, axis=0)
    
    @Output()
    def Factor_GTJA_15(self):
        return self.Data['Open']/self.Data['Close'].shift(axis=0)
    
    @Output()
    def Factor_GTJA_16(self):
        return -self.TSCorr(self.Data['Volume'].rank(axis=1, pct=True), 
                            self.Data['VWAP'].rank(axis=1, pct=True), 5).rank(
                                    axis=1, pct=True).rolling(5, axis=0).max()
    
    @Output()
    def Factor_GTJA_17(self):
        return (self.Data['VWAP'] -
                self.Data['VWAP'].rolling(15, axis=0).max()).rank(axis=1, 
                         pct=True)**(self.Data['Close'].pct_change(5, axis=0))
    
    @Output()
    def Factor_GTJA_18(self):
        return self.Data['Close'].pct_change(5, axis=0)
    
    @Output()
    def Factor_GTJA_19(self):
        diff = self.Data['Close'].diff(axis=0)
        div = self.Data['Close'].shift(axis=0).where(diff>=0, 
                       self.Data['Close'])
        return diff / div
    
    @Output()
    def Factor_GTJA_20(self):
        return self.Data['Close'].pct_change(6, axis=0)
    
    @Output()
    def Factor_GTJA_21(self):
        return self.RegBeta(self.Data['Close'], np.arange(6)+1)
    
    @Output()
    def Factor_GTJA_22(self):
        return (self.Data['Close']/self.Data['Close'].rolling(
                6, axis=0).mean() - 1).diff(3, axis=0).ewm(
            alpha=1/12, axis=0).mean()
    
    @Output()
    def Factor_GTJA_23(self):
        temp = (self.Data['Close'].rolling(20, axis=0).std() * 
                (self.Data['Close'].diff(axis=0)>0)).ewm(alpha=1/20,
                axis=0).mean() 
        return 1 - (self.Data['Close'].rolling(20, axis=0).std() * 
                    (self.Data['Close'].diff(axis=0)<=0)).ewm(alpha=1/20,
                    axis=0).mean() / temp 
     
    @Output()
    def Factor_GTJA_24(self):
        return self.Data['Close'].diff(5, axis=0).ewm(alpha=0.2, axis=0).mean()
        
    @Output()
    def Factor_GTJA_25(self):
        return -(self.Data['Close'].diff(7, axis=0) *
                 (1 - (self.Data['Volume'] /
                       self.Data['Volume'].rolling(20,
                                axis=0).mean()).rolling(9,
                                axis=0).apply(lambda x: 
                                    self.DecayLinear(x)).rank(axis=1,
                                                    pct=True))).rank(
                           axis=1) * (1+self.Data['Close'].pct_change(
                                   axis=0).rolling(250,
                                         axis=0).sum().rank(axis=1, pct=True))
    
    @Output()
    def Factor_GTJA_26(self):
        return (self.Data['Close'].rolling(7, axis=0).mean() -
                self.Data['Close'] + self.TSCorr(self.Data['VWAP'],
                         self.Data['Close'].shift(5, axis=0), 230))
        
    @Output()
    def Factor_GTJA_27(self):
        return (self.Data['Close'].pct_change(3, axis=0)*100 +
                self.Data['Close'].pct_change(6, axis=0)*100).rolling(12,
                         axis=0).apply(lambda x: self.DecayLinear(x))
    
    @Output()
    def Factor_GTJA_28(self):
        lmin = self.Data['Low'].rolling(9, axis=0).min()
        return 3*((self.Data['Close'] - lmin) / 
                  (self.Data['High'].rolling(9, axis=0).max() - lmin)*
                  100).ewm(alpha=1/3, axis=0).mean() - 2*(
                          (self.Data['Close'] - lmin)/
                          (self.Data['High'].rolling(9, axis=0).max() -
                           self.Data['Low'].rolling(9, axis=0).max())*
                           100).ewm(alpha=1/3, axis=0).mean().ewm(
                                   alpha=1/3, axis=0).mean()
    
    @Output()
    def Factor_GTJA_29(self):
        return self.Data['Close'].pct_change(6, axis=0) * self.Data['Volume']
    
    @Output()
    def Factor_GTJA_30(self):
        pass
    
    @Output()
    def Factor_GTJA_31(self):
        return (self.Data['Close'] / 
                self.Data['Close'].rolling(12, axis=0).mean() - 1)
    
    @Output()
    def Factor_GTJA_32(self):
        return -self.TSCorr(self.Data['High'].rank(axis=1, pct=True),
                            self.Data['Volume'].rank(axis=1, pct=True),
                            3).rank(axis=1, pct=True).rolling(3, axis=0).sum()
    
    @Output()
    def Factor_GTJA_33(self):
        ret = self.Data['Close'].pct_change(axis=0)
        return -(self.Data['Low'].rolling(5, axis=0).min().diff(5, axis=0) * 
                 ((ret.rolling(240, axis=0).sum() - 
                   ret.rolling(20, axis=0).sum()) / 220).rank(axis=1, 
                   pct=True) * self.TSRank(self.Data['Volume'], 5))
    
    @Output()
    def Factor_GTJA_34(self):
        return (self.Data['Close'] /
                self.Data['Close'].rolling(12, axis=1).mean())
    
    @Output()
    def Factor_GTJA_35(self):
        return np.minimum(self.Data['Open'].diff(axis=0).rolling(15,
                          axis=0).apply(lambda x:
                              self.DecayLinear(x)).rank(axis=1, pct=True),
                          self.TSCorr(self.Data['Volume'],
                                      self.Data['Open'], 17).rolling(7,
                                               axis=0).apply(lambda x: 
                                           self.DecayLinear(x)).rank(axis=1, 
                                                           pct=True))
            
    @Output()
    def Factor_GTJA_36(self):
        return self.TSCorr(self.Data['Volume'].rank(axis=1, pct=True), 
                           self.Data['VWAP'].rank(axis=1, pct=True),
                           6).rolling(2, axis=0).sum().rank(axis=1, pct=True)
    
    @Output()
    def Factor_GTJA_37(self):
        return -(self.Data['Close'].pct_change(axis=0).rolling(5, 
                 axis=0).sum() -
            self.Data['Open'].rolling(5, axis=0).sum()).diff(10, 
                     axis=0).rank(axis=1, pct=True)
        
    @Output()
    def Factor_GTJA_38(self):
        return ((-self.Data['High'].diff(2, axis=0)) * 
                (self.Data['High'].rolling(20, axis=0).mean() < 
                 self.Data['High']))
    
    @Output()
    def Factor_GTJA_39(self):
        return (self.Data['Close'].diff(2, axis=0).rolling(8, 
                        axis=0).apply(lambda x: 
                            self.DecayLinear(x)).rank(axis=1, pct=True) +
            self.TSCorr(0.3*self.Data['VWAP'] +
                        0.7*self.Data['Open'],
                        self.Data['Volume'].rolling(180,
                                 axis=0).mean().rolling(37, axis=0).sum(), 
                                 14).rolling(12, axis=0).apply(lambda x:
                                     self.DecayLinear(x)).rank(axis=1, 
                                                     pct=True))
    @Output()
    def Factor_GTJA_40(self):
        return ((self.Data['Volume'] *
                (self.Data['Close'].diff(axis=0)>0)).rolling(26, 
                axis=0).sum() /
                (self.Data['Volume'] *
                 (self.Data['Close'].diff(axis=0)<=0)).rolling(26,
                 axis=0).sum())
    
    @Output()
    def Factor_GTJA_41(self):
        return -self.Data['VWAP'].diff(3, axis=0).rolling(5,
                         axis=0).max().rank(axis=1, pct=True)
    
    @Output()
    def Factor_GTJA_42(self):
        return ((-self.Data['High'].rolling(10, axis=0).std().rank(axis=1,
                 pct=True)) * self.TSCorr(self.Data['High'], 
            self.Data['Volume'], 10))
    
    @Output()
    def Factor_GTJA_43(self):
        return (self.Data['Volume'] *
                (2*((self.Data['Close'].diff(axis=0)>0) -
                    0.5))).rolling(6, axis=0).sum()
    
    @Output()
    def Factor_GTJA_44(self):
        return (self.TSRank(self.TSCorr(self.Data['Low'],
                            self.Data['Volume'].rolling(10,
                                     axis=0).mean(), 7).rolling(6,
                                     axis=0).apply(lambda x:
                                         self.DecayLinear(x)), 4) +
                                self.TSRank(self.Data['VWAP'].diff(3,
                                 axis=0).rolling(10, 
                                 axis=0).apply(lambda x:
                                 self.DecayLinear(x)), 15))
    
    @Output()
    def Factor_GTJA_45(self):
        return ((0.6*self.Data['Close'] +
                 0.4*self.Data['Open']).diff(axis=0).rank(axis=1, pct=True) *
                    self.TSCorr(self.Data['VWAP'],
                                self.Data['Volume'].rolling(150, 
                                         axis=0).mean(),
                                         15).rank(axis=1, pct=True))
    
    @Output()
    def Factor_GTJA_46(self):
        return ((self.Data['Close'].rolling(3, axis=0).mean() +
                 self.Data['Close'].rolling(6, axis=0).mean() +
                 self.Data['Close'].rolling(12, axis=0).mean() +
                 self.Data['Close'].rolling(24, axis=0).mean()) /
                    self.Data['Close'])
    
    @Output()
    def Factor_GTJA_47(self):
        h_max = self.Data['High'].rolling(6, axis=0).max()
        return ((h_max - self.Data['Close']) /
                (h_max - self.Data['Low'].rolling(6, axis=0).min()) *
                100).ewm(alpha=1/9, axis=0).mean()
    
    @Output()
    def Factor_GTJA_48(self):
        diff = 2*((self.Data['Close'].diff(axis=0) > 0) - 0.5)
        return -(diff.rolling(3, axis=0).sum().rank(axis=1, pct=True) * 
                 self.Data['Volume'].rolling(5, axis=0).sum() / 
                 self.Data['Volume'].rolling(20, axis=0).sum())
    
    @Output()
    def Factor_GTJA_49(self):
        diff_h = self.Data['High'].diff(axis=0)
        diff_l = self.Data['Low'].diff(axis=0)
        cond1 = diff_h >= diff_l
        cond2 = diff_h <= diff_l
        temp = np.maximum(self.Data['High'].diff(axis=0).abs(),
                          self.Data['Low'].diff(axis=0).abs())
        s1 = (cond1*temp).rolling(12, axis=0).sum()
        s2 = (cond2*temp).rolling(12, axis=0).sum()
        return s1 / (s1+s2)
    
    @Output()
    def Factor_GTJA_50(self):
        diff_h = self.Data['High'].diff(axis=0)
        diff_l = self.Data['Low'].diff(axis=0)
        cond1 = diff_h >= diff_l
        cond2 = diff_h <= diff_l
        temp = np.maximum(self.Data['High'].diff(axis=0).abs(),
                          self.Data['Low'].diff(axis=0).abs())
        s1 = (cond1*temp).rolling(12, axis=0).sum()
        s2 = (cond2*temp).rolling(12, axis=0).sum()
        return (s1-s2) / (s1+s2)
    
    @Output()
    def Factor_GTJA_51(self):
        diff_h = self.Data['High'].diff(axis=0)
        diff_l = self.Data['Low'].diff(axis=0)
        cond1 = diff_h >= diff_l
        cond2 = diff_h <= diff_l
        temp = np.maximum(self.Data['High'].diff(axis=0).abs(),
                          self.Data['Low'].diff(axis=0).abs())
        s1 = (cond1*temp).rolling(12, axis=0).sum()
        s2 = (cond2*temp).rolling(12, axis=0).sum()
        return s2 / (s1+s2)
    
    @Output()
    def Factor_GTJA_52(self):
        m_3 = (self.Data['High']+self.Data['Low']+
               self.Data['Close']).shift(axis=0)/3
        temp1 = self.Data['High'] - m_3
        temp1[temp1 < 0] = 0
        temp2 = m_3 - self.Data['Low']
        temp2[temp2 < 0] = 0
        return temp1.rolling(26, axis=0).sum()/temp2.rolling(26, axis=0).sum()
        
    @Output()
    def Factor_GTJA_53(self):
        return (self.Data['Close'].diff(axis=0) > 0).rolling(12,
               axis=0).sum() / 12
    
    @Output()
    def Factor_GTJA_54(self):
        temp = self.Data['Close'] - self.Data['Open']
        return -(temp.abs().rolling(10, axis=0).std() +
                 temp +
                 self.TSCorr(self.Data['Close'], 
                             self.Data['Open'], 10)).rank(axis=1, pct=True)
    
    @Output()
    def Factor_GTJA_55(self):
        pass
    
    @Output()
    def Factor_GTJA_56(self):
        return ((self.TSCorr(((self.Data['High'] +
                               self.Data['Low'])/2).rolling(19, axis=0).sum(),
                self.Data['Volume'].rolling(40, axis=0).mean().rolling(19,
                     axis=0).sum(), 13).rank(axis=1,
                     pct=True)**5).rank(axis=1, pct=True) -
                     (self.Data['Open'] -
                      self.Data['Open'].rolling(12, 
                               axis=0).min()).rank(axis=1, pct=True))
    
    @Output()
    def Factor_GTJA_57(self):
        min_l = self.Data['Low'].rolling(9, axis=0).min()
        return ((self.Data['Close'] - min_l) /
                (self.Data['High'].rolling(9, axis=0).max() -
                 min_l)*100).ewm(alpha=1/3, axis=0).mean()
    
    @Output()
    def Factor_GTJA_58(self):
        return (self.Data['Close'].diff(axis=0)>0).rolling(20, axis=0).sum()
    
    @Output()
    def Factor_GTJA_59(self):
        s_c = self.Data['Close'].shift(axis=0)
        d_c = self.Data['Close'].diff(axis=0)
        return (self.Data['Close'] -
                np.minimum(self.Data['Low'], s_c)*(d_c > 0) -
                np.maximum(self.Data['High'], s_c)*(d_c < 0)).rolling(20,
                          axis=0).sum()
    
    @Output()
    def Factor_GTJA_60(self):
        return ((self.Data['Close']*2 - self.Data['Low'] - self.Data['High'])/
                (self.Data['High'] - self.Data['Low'])*
                self.Data['Volume']).rolling(20, axis=0).sum()  
    
    @Output()
    def Factor_GTJA_61(self):
        temp1 = self.Data['VWAP'].diff(axis=0).rolling(12,
                         axis=0).apply(lambda x:
                             self.DecayLinear(x)).rank(axis=1, pct=True)
        temp2 = self.TSCorr(self.Data['Low'], self.Data['Volume'].rolling(80, 
                            axis=0).mean(), 8).rank(axis=1,
            pct=True).rolling(17, axis=0).apply(lambda x:
                self.DecayLinear(x)).rank(axis=1, pct=True)
        return -np.maximum(temp1, temp2)
        
    @Output()
    def Factor_GTJA_62(self):
        return -self.TSCorr(self.Data['High'],
                            self.Data['Volume'].rank(axis=1, pct=True),
                            5)
    
    @Output()
    def Factor_GTJA_63(self):
        diff = self.Data['Close'].diff(axis=0)
        p2 = diff.abs().ewm(alpha=1/6, axis=0).mean()
        diff[diff < 0] = 0
        p1 = diff.ewm(alpha=1/6, axis=0).mean()
        return -p1/p2*100
    
    @Output()
    def Factor_GTJA_64(self):
        temp1 = self.TSCorr(self.Data['VWAP'].rank(axis=1, pct=True), 
                            self.Data['Volume'].rank(axis=1, pct=True),
                            4).rolling(4, axis=0).apply(lambda x:
                                self.DecayLinear(x)).rank(axis=1, pct=True)
        temp2 = self.TSCorr(self.Data['Close'].rank(axis=1, pct=True), 
                            self.Data['Volume'].rolling(60,
                                     axis=0).mean().rank(axis=1, pct=True),
                                     4).rolling(13, axis=0).max().rolling(14,
                                               axis=0).apply(lambda x: 
                                   self.DecayLinear(x)).rank(axis=1, pct=True)
        return np.maximum(temp1, temp2)
    
    @Output()
    def Factor_GTJA_65(self):
        return self.Data['Close'].rolling(6, axis=0).mean()/self.Data['Close']
        
    @Output()
    def Factor_GTJA_66(self):
        return (self.Data['Close'] /
                self.Data['Close'].rolling(6, axis=0).mean() - 1)
    
    @Output()
    def Factor_GTJA_67(self):
        diff = self.Data['Close'].diff(axis=0)
        p2 = diff.abs().ewm(alpha=1/24, axis=0).mean()
        diff[diff < 0] = 0
        p1 = diff.ewm(alpha=1/24, axis=0).mean()
        return p1/p2*100
    
    @Output()
    def Factor_GTJA_68(self):
        return ((self.Data['High'].diff(axis=0) + 
                 self.Data['Low'].diff(axis=0))*
                (self.Data['High'] -
                 self.Data['Low'])/self.Data['Volume']).ewm(alpha=2/15,
                 axis=0).mean()
    
    @Output()
    def Factor_GTJA_69(self):
        d_o = self.Data['Open'].diff(axis=0)
        dtm = np.maximum(self.Data['High']-self.Data['Open'], d_o)*(d_o>0)
        dbm = np.maximum(self.Data['Open']-self.Data['Low'], d_o)*(d_o<0)
        s_dtm = dtm.rolling(20, axis=0).sum()
        s_dbm = dbm.rolling(20, axis=0).sum()
        return (1-s_dbm/s_dtm)*(s_dtm>s_dbm) + (s_dtm/s_dbm-1)*(s_dtm<s_dbm)
        
    @Output()
    def Factor_GTJA_70(self):
        return self.Data['Amount'].rolling(6, axis=0).std()
    
    @Output()
    def Factor_GTJA_71(self):
        return (self.Data['Close'] / 
                self.Data['Close'].rolling(24, axis=0).mean() - 1)
    
    @Output()
    def Factor_GTJA_72(self):
        h_max = self.Data['High'].rolling(6, axis=0).max()
        return ((h_max - self.Data['Close'])/
                (h_max - self.Data['Low'].rolling(6,
                 axis=0).min())).ewm(alpha=1/15, axis=0).mean()
    
    @Output()
    def Factor_GTJA_73(self):
        return -(self.TSRank(self.TSCorr(self.Data['Close'], 
                             self.Data['Volume'],
                             10).rolling(16, axis=0).apply(lambda x:
                                 self.DecayLinear(x)).rolling(4,
                                                 axis=0).apply(lambda x:
                     self.DecayLinear(x)), 5) -
            self.TSCorr(self.Data['VWAP'], self.Data['Volume'].rolling(30, 
                        axis=0).mean(), 4).rolling(3, axis=0).apply(lambda x:
            self.DecayLinear(x)).rank(axis=1, pct=True))
    
    @Output()
    def Factor_GTJA_74(self):
        return self.TSCorr((self.Data['Low']*0.35 + 
                            self.Data['VWAP']*0.65).rolling(20,
                             axis=0).sum(), self.Data['Volume'].rolling(40, 
                                        axis=0).mean().rolling(20,
                                        axis=0).sum(), 7).rank(axis=1,
            pct=True) + self.TSCorr(self.Data['VWAP'].rank(axis=1,
                    pct=True), self.Data['Volume'].rank(axis=1,
                            pct=True), 6).rank(axis=1, pct=True)
    
    @Output()
    def Factor_GTJA_75(self):
        return (((self.Data['Close']>self.Data['Open']).mul(
            self.Data['Index']<self.Data['IndexOpen'], axis=0)).rolling(50,
                 axis=0).sum().div( 
                (self.Data['Index']<self.Data['IndexOpen']).rolling(50, 
                axis=0).sum(), axis=0))
    
    @Output()
    def Factor_GTJA_76(self):
        temp = (self.Data['Close'].pct_change(axis=0) /
                self.Data['Volume']).rolling(20, axis=0)
        return temp.std() / temp.mean()
    
    @Output()
    def Factor_GTJA_77(self):
        temp1 = ((self.Data['High'] + self.Data['Low'])/2 - 
                 self.Data['VWAP']).rolling(20, axis=0).apply(lambda x: 
                     self.DecayLinear(x)).rank(axis=1, pct=True)
        temp2 = self.TSCorr((self.Data['High'] + self.Data['Low'])/2, 
                            self.Data['Volume'].rolling(40,
                                     axis=0).mean(), 3).rolling(6,
                                     axis=0).apply(lambda x:
                                 self.DecayLinear(x)).rank(axis=1, pct=True)
        return np.minimum(temp1, temp2)
    
    @Output()
    def Factor_GTJA_78(self):
        temp = (self.Data['High'] + self.Data['Low'] + self.Data['Close'])/3
        return ((temp - temp.rolling(12, axis=0).mean()) /
                (0.015*(self.Data['Close'] - 
                        temp.rolling(12, axis=0).mean()).abs().rolling(12,
                                    axis=0).mean()))
    
    @Output()
    def Factor_GTJA_79(self):
        temp = self.Data['Close'].diff(axis=0)
        return (temp.rolling(10, axis=0).max().ewm(alpha=1/12, axis=0).mean()/
                temp.abs().ewm(alpha=1/12, axis=0).mean())
    
    @Output()
    def Factor_GTJA_80(self):
        return self.Data['Volume'].pct_change(5, axis=0)*100
    
    @Output()
    def Factor_GTJA_81(self):
        return self.Data['Volume'].ewm(alpha=2/21).mean()
    
    @Output()
    def Factor_GTJA_82(self):
        h_max = self.Data['High'].rolling(6, axis=0).max()
        return ((h_max - self.Data['Close'])/
                (h_max - self.Data['Low'].rolling(6,
                 axis=0).min())).ewm(alpha=1/20, axis=0).mean()
    
    @Output()
    def Factor_GTJA_83(self):
        return -(self.TSCov(self.Data['High'].rank(axis=1, pct=True),
                            self.Data['Volume'].rank(axis=1, pct=True),
                            5).rank(axis=1, pct=True))
    
    @Output()
    def Factor_GTJA_84(self):
        return (self.Data['Volume'] *
                (self.Data['Close'].diff(axis=0)>0) -
                (self.Data['Volume'] *
                 (self.Data['Close'].diff(axis=0)<=0))).rolling(20,
                 axis=0).sum()
    
    @Output()
    def Factor_GTJA_85(self):
        return (self.TSRank(self.Data['Volume']/
                            self.Data['Volume'].rolling(20, axis=0).mean(),
                            20) *
                self.TSRank(-self.Data['Close'].diff(7, axis=0), 8))
    
    @Output()
    def Factor_GTJA_86(self):
        d_10 = self.Data['Close'].shift(10, axis=0)
        d_20 = self.Data['Close'].shift(20, axis=0)
        temp = (d_20 - d_10 - d_10 + self.Data['Close'])/10
        res = self.Data['Close'].diff(axis=0)
        res[temp > 0.25] = -1.0
        res[temp < 0] = 1.0
        return res
    
    @Output()
    def Factor_GTJA_87(self):
        return (self.Data['VWAP'].diff(4, axis=0).rolling(7,
                axis=0).apply(lambda x:
                    self.DecayLinear(x)).rank(axis=1, pct=True) -
            self.TSRank(((self.Data['Low'] - self.Data['VWAP'])/
                         (self.Data['Open'] -
                      (self.Data['High'] + self.Data['Low'])/2)).rolling(11,
                        axis=0).apply(lambda x: self.DecayLinear(x)), 7))
    
    @Output()
    def Factor_GTJA_88(self):
        return self.Data['Close'].pct_change(20, axis=0)*100
    
    @Output()
    def Factor_GTJA_89(self):
        ema1 = self.Data['Close'].ewm(alpha=2/13, axis=0).mean()
        ema2 = self.Data['Close'].ewm(alpha=2/27, axis=0).mean()
        temp = ema1 - ema2
        return 2*(temp - temp.ewm(alpha=0.2, axis=0).mean())
    
    @Output()
    def Factor_GTJA_90(self):
        return -self.TSCorr(self.Data['VWAP'].rank(axis=1, pct=True), 
                            self.Data['Volume'].rank(axis=1, pct=True),
                            5).rank(axis=1, pct=True)

    @Output()
    def Factor_GTJA_91(self):
        return -((self.Data['Close'] -
                  self.Data['Close'].rolling(5, axis=0).max()).rank(axis=1,
                           pct=True) * 
            self.TSCorr(self.Data['Volume'].rolling(40, axis=0).mean(), 
                        self.Data['Low'], 5).rank(axis=1, pct=True))
       
    @Output()
    def Factor_GTJA_92(self):
        r1 = (self.Data['Close']*0.35 +
              self.Data['VWAP']*0.65).diff(2, axis=0).rolling(3,
                       axis=0).apply(lambda x:
                           self.DecayLinear(x)).rank(axis=1, pct=True)
        r2 = self.TSRank(self.TSCorr(self.Data['Volume'].rolling(180,
                                     axis=0).mean(),
            self.Data['Close'], 13).abs().rolling(5, axis=0).apply(lambda x:
                self.DecayLinear(x)), 5)
        return -np.maximum(r1, r2)
    
    @Output()
    def Factor_GTJA_93(self):
        diff = self.Data['Open'].diff(axis=0)
        return (np.maximum(self.Data['Open'] - 
                           self.Data['Low'], diff) *
            (diff < 0)).rolling(20, axis=0).sum()
    
    @Output()
    def Factor_GTJA_94(self):
        return (self.Data['Volume'] *
                (self.Data['Close'].diff(axis=0)>0) -
                (self.Data['Volume'] *
                 (self.Data['Close'].diff(axis=0)<=0))).rolling(30,
                 axis=0).sum()
            
    @Output()
    def Factor_GTJA_95(self):
        return self.Data['Amount'].rolling(20, axis=0).std()
    
    @Output()
    def Factor_GTJA_96(self):
        l_min = self.Data['Low'].rolling(9, axis=0).min()
        return ((self.Data['Close'] - l_min)/
                (self.Data['High'] - l_min)*100).ewm(alpha=1/3,
                axis=0).mean().ewm(alpha=1/3, axis=0).mean()
        
    @Output()
    def Factor_GTJA_97(self):
        return self.Data['Volume'].rolling(10, axis=0).std()
    
    @Output()
    def Factor_GTJA_98(self):
        res = -self.Data['Close'].diff(3, axis=0)
        temp = (self.Data['Close'].rolling(100,
                axis=0).mean().diff(100, axis=0) / 
                self.Data['Close'].shift(100, axis=0))
        res[temp <= 0.05] = (self.Data['Close'].rolling(100, axis=0).min() -
                             self.Data['Close'])
        return res
    
    @Output()
    def Factor_GTJA_99(self):
        return -(self.TSCov(self.Data['Close'].rank(axis=1, pct=True),
                            self.Data['Volume'].rank(axis=1, pct=True),
                            5).rank(axis=1, pct=True))
    
    @Output()
    def Factor_GTJA_100(self):
        return self.Data['Volume'].rolling(20, axis=0).std()
    
    @Output()
    def Factor_GTJA_101(self):
        return (-1*(self.TSCorr(self.Data['Close'], 
                                self.Data['Volume'].rolling(30, 
                                         axis=0).mean().rolling(37, 
                             axis=0).sum(), 15).rank(axis=1, pct=True) <
            self.TSCorr((self.Data['High']*0.1 +
                         self.Data['VWAP']*0.9).rank(axis=1, pct=True),
            self.Data['Volume'].rank(axis=1, pct=True),
            11).rank(axis=1, pct=True))).rolling(10, axis=0).sum()
    
    @Output()
    def Factor_GTJA_102(self):
        return (self.Data['Volume'].diff(axis=0).rolling(10, 
                        axis=0).max().ewm(alpha=1/6, axis=0).mean() / 
                        self.Data['Volume'].diff(axis=0).abs().ewm(alpha=1/6,
                                 axis=0).mean())
    
    @Output()
    def Factor_GTJA_103(self):
        return 1 - self.Data['Low'].rolling(20, axis=0).apply(self.LowDay)/20
    
    @Output()
    def Factor_GTJA_104(self):
        return -(self.TSCorr(self.Data['High'],
                             self.Data['Volume'], 5).diff(5, axis=0) *
            self.Data['Close'].rolling(20,
                     axis=0).std().rank(axis=1, pct=True))
    
    @Output()
    def Factor_GTJA_105(self):
        return -(self.TSCorr(self.Data['Open'].rank(axis=1, pct=True), 
                             self.Data['Volume'].rank(axis=1, pct=True),
                             10))
    
    @Output()
    def Factor_GTJA_108(self):
        return -((self.Data['High'] -
                  self.Data['High'].rolling(2, axis=0).min()).rank(axis=1,
                           pct=True) ** 
                     (self.TSCorr(self.Data['VWAP'],
                                  self.Data['Volume'].rolling(120,
                                       axis=0).mean(),
                                        6).rank(axis=1, pct=True)))
    
    @Output()
    def Factor_GTJA_110(self):
        s_c = self.Data['Close'].shift(axis=0)
        temp1 = self.Data['High'] - s_c
        temp2 = s_c - self.Data['Low']
        temp1[temp1 < 0] = 0.0
        temp2[temp2 < 0] = 0.0
        return (temp1.rolling(20, axis=0).sum() /
                temp2.rolling(20, axis=0).sum())
    
    @Output()
    def Factor_GTJA_111(self):
        temp = ((self.Data['Close']*2 - self.Data['Low'] - self.Data['High']) *
                self.Data['Volume']/(self.Data['High'] - self.Data['Low']))
        return (temp.ewm(alpha=2/11, axis=0).mean() /
                temp.ewm(alpha=0.5, axis=0).mean())
    
    @Output()
    def Factor_GTJA_113(self):
        return -(self.Data['Close'].shift(5, axis=0).rolling(20, 
                 axis=0).mean().rank(axis=1, pct=True) *
            self.TSCorr(self.Data['Close'],
                        self.Data['Volume'], 2) *
                        self.TSCorr(self.Data['Close'].rolling(5,
                                    axis=0).sum(), 
            self.Data['Close'].rolling(20,
                     axis=0).sum(), 2).rank(axis=1, pct=True))
    
    @Output()
    def Factor_GTJA_114(self):
        temp = ((self.Data['High'] - self.Data['Low'])/
                self.Data['Close'].rolling(5, axis=0).mean())
        return (temp.shift(2, axis=0).rank(axis=1, pct=True) *
                self.Data['Volume'].rank(axis=1,
                         pct=True).rank(axis=1, pct=True) /
                         (temp/(self.Data['VWAP'] - self.Data['Close'])))
    
    @Output()
    def Factor_GTJA_115(self):
        return (self.TSCorr(self.Data['High']*0.9 +
                            self.Data['Close']*0.1,
                            self.Data['Volume'].rolling(30, axis=0).mean(),
                            10) ** 
                self.TSCorr(self.TSRank((self.Data['High'] + 
                                         self.Data['Low'])/2, 4),
                self.TSRank(self.Data['Volume'], 10),
                7).rank(axis=1, pct=True))
        
    @Output()
    def Factor_GTJA_117(self):
        return (self.TSRank(self.Data['Volume'], 32) * 
                (1 - self.TSRank(self.Data['Close']+self.Data['High']-
                                 self.Data['Low'], 16)) * 
                (1 - self.TSRank(self.Data['Close'].pct_change(axis=0), 32)))
    
    @Output()
    def Factor_GTJA_118(self):
        return ((self.Data['High'] - 
                 self.Data['Open']).rolling(20, axis=0).sum() /
                    (self.Data['Open'] - 
                     self.Data['Low']).rolling(20, axis=0).sum())
        
    @Output()
    def Factor_GTJA_119(self):
        return (self.TSCorr(self.Data['VWAP'],
                            self.Data['Volume'].rolling(5,
                                     axis=0).mean().rolling(26,
                                     axis=0).sum(), 5).rolling(7,
                                     axis=0).apply(lambda x:
               self.DecayLinear(x)).rank(axis=1, pct=True) -
               (self.TSRank(self.TSCorr(self.Data['Open'].rank(axis=1,
                                        pct=True), 
                       self.Data['Volume'].rolling(15, 
                                axis=0).mean().rank(axis=1,
                                pct=True), 21).rolling(9,
                                axis=0).min(), 7).rolling(8,
                                axis=0).apply(lambda x: 
                            self.DecayLinear(x))).rank(axis=1, pct=True))
                           
    @Output()
    def Factor_GTJA_120(self):
        r1 = (self.Data['VWAP'] - self.Data['Close']).rank(axis=1, pct=True)
        r2 = (self.Data['VWAP'] + self.Data['Close']).rank(axis=1, pct=True)
        return r1 / r2
                    
    @Output()
    def Factor_GTJA_121(self):
        bot = (self.Data['VWAP'] -
               self.Data['VWAP'].rolling(12, 
                        axis=0).min()).rank(axis=1, pct=True)
        nor = self.TSRank(self.TSCorr(self.TSRank(self.Data['VWAP'], 20),
                                  self.TSRank(self.Data['Volume'].rolling(60,
                                              axis=0).mean(), 2), 18), 3)
        return bot ** nor
    
    @Output()
    def Factor_GTJA_122(self):
        return (np.log(self.Data['Close']).ewm(alpha=2/13, 
                axis=0).mean().ewm(alpha=2/13,
                            axis=0).mean().ewm(alpha=2/13, 
                                        axis=0).mean().pct_change(axis=0))
    
    @Output()
    def Factor_GTJA_123(self):
        return (self.TSCorr(((self.Data['High'] +
                              self.Data['Low'])/2).rolling(20,
            axis=0).sum(),
            self.Data['Volume'].rolling(60, axis=0).mean().rolling(20,
                     axis=0).sum(), 9).rank(axis=1, pct=True) -
            self.TSCorr(self.Data['Low'],
                        self.Data['Volume'], 6).rank(axis=1, pct=True))
    
    @Output()
    def Factor_GTJA_124(self):
        return ((self.Data['Close'] - self.Data['VWAP']) /
                self.Data['Close'].rolling(30, axis=0).max().rank(axis=1,
                         pct=True).rolling(2, axis=0).apply(lambda x:
                             self.DecayLinear(x)))
            
    @Output()
    def Factor_GTJA_125(self):
        r1 = (self.TSCorr(self.Data['VWAP'],
                          self.Data['Volume'].rolling(80, 
                                   axis=0).mean(), 17).rolling(20,
                                   axis=0).apply(lambda x: 
                               self.DecayLinear(x)).rank(axis=1, pct=True))
        r2 = (((self.Data['Close'] + 
                self.Data['VWAP'])/2).diff(3, axis=0).rolling(16,
                axis=0).apply(lambda x:
                    self.DecayLinear(x)).rank(axis=1, pct=True))
        return r1 / r2
    
    @Output()
    def Factor_GTJA_126(self):
        return (self.Data['Close'] + self.Data['High'] + self.Data['Low'])/3
    
    @Output()
    def Factor_GTJA_127(self):
        m_c = self.Data['Close'].rolling(12, axis=0).max()
        return np.sqrt((((self.Data['Close']/m_c
                          - 1)*100)**2).rolling(12, axis=0).mean())
    
    @Output()
    def Factor_GTJA_128(self):
        temp = (self.Data['Close'] + self.Data['High'] + self.Data['Low'])/3
        res = temp*self.Data['Volume']
        res1 = res.copy()
        res1[temp.diff(axis=0)<0] = 0
        res2 = res.copy()
        res2[temp.diff(axis=0)>0] = 0
        return (100 - (100/(1 + res1.rolling(14, axis=0).sum() /
                            res2.rolling(14, axis=0).sum())))
        
    @Output()
    def Factor_GTJA_129(self):
        temp = -self.Data['Close'].diff(axis=0)
        temp[temp < 0] = 0
        return temp.rolling(12, axis=0).sum()
    
    @Output()
    def Factor_GTJA_130(self):
        r1 = (self.TSCorr((self.Data['High'] + self.Data['Low'])/2, 
                          self.Data['Volume'].rolling(40, 
                                   axis=0).mean(), 9).rolling(10, 
                                   axis=0).apply(lambda x: 
                       self.DecayLinear(x)).rank(axis=1, pct=True))
        r2 = (self.TSCorr(self.Data['VWAP'].rank(axis=1, pct=True), 
                          self.Data['Volume'].rank(axis=1,
                                   pct=True), 7).rolling(3, 
                                   axis=0).apply(lambda x: 
                       self.DecayLinear(x)).rank(axis=1, pct=True))
        return r1 / r2

    @Output()
    def Factor_GTJA_131(self):
        bot = self.Data['VWAP'].diff(axis=0).rank(axis=1, pct=True)
        nor = self.TSRank(self.TSCorr(self.Data['Close'], 
                                      self.Data['Volume'].rolling(50,
                                               axis=0).mean(), 18), 18)
        return bot ** nor
    
    @Output()
    def Factor_GTJA_132(self):
        return self.Data['Amount'].rolling(20, axis=0).mean()
    
    @Output()
    def Factor_GTJA_133(self):
        return ((20 - self.Data['High'].rolling(20,
                axis=0).apply(self.HighDay))/20 - 
                (20 - self.Data['Low'].rolling(20,
                 axis=0).apply(self.LowDay))/20)
    
    @Output()
    def Factor_GTJA_134(self):
        return self.Data['Volume']*self.Data['Close'].pct_change(12, axis=0)
    
    @Output()
    def Factor_GTJA_135(self):
        return (self.Data['Close'] / self.Data['Close'].shift(20, 
                         axis=0)).shift(axis=0).ewm(alpha=0.05,
                         axis=0).mean()
        
    @Output()
    def Factor_GTJA_136(self):
        return -(self.TSCorr(self.Data['Open'],
                             self.Data['Volume'], 10) * 
            self.Data['Close'].pct_change(axis=0).diff(3, axis=0))
        
    @Output()
    def Factor_GTJA_137(self):
        pass
    
    @Output()
    def Factor_GTJA_138(self):
        r1 = (self.Data['Low']*0.7 +
              self.Data['VWAP']*0.3).diff(3, axis=0).rolling(20,
                       axis=0).apply(lambda x:
                           self.DecayLinear(x)).rank(axis=1, pct=True)
            
        r2 = self.TSRank(self.TSCorr(self.TSRank(self.Data['Close'], 8), 
                                 self.TSRank(self.Data['Volume'].rolling(60, 
                                             axis=0).mean(), 20),
                                     8).rolling(7, axis=0).apply(lambda x:
                                         self.DecayLinear(x)), 3)    
        return np.minimum(r1, r2)
    
    @Output()
    def Factor_GTJA_139(self):
        return -self.TSCorr(self.Data['Open'], self.Data['Volume'], 10)
    
    @Output()
    def Factor_GTJA_140(self):
        r1 = (self.Data['Open'].rank(axis=1, pct=True) +
              self.Data['Low'].rank(axis=1, pct=True) - 
              self.Data['High'].rank(axis=1, pct=True) -
              self.Data['Close'].rank(axis=1, pct=True)).rolling(8, 
                       axis=0).apply(lambda x: 
                           self.DecayLinear(x)).rank(axis=1, pct=True)
        r2 = self.TSRank(self.TSCorr(self.TSRank(self.Data['Close'], 8),
                                     self.TSRank(self.Data['Volume'], 60),
                                     20).rolling(8, axis=0).apply(lambda x: 
                                         self.DecayLinear(x)), 3)
        return np.minimum(r1, r2)
    
    @Output()
    def Factor_GTJA_141(self):
        return -self.TSCorr(self.Data['High'].rank(axis=1, pct=True), 
                            self.Data['Volume'].rolling(15, axis=0).mean(),
                            9)
        
    @Output()
    def Factor_GTJA_142(self):
        return (self.TSRank(self.Data['Close'], 10).rank(axis=1, pct=True) *
                self.Data['Close'].diff(axis=0).diff(axis=0).rank(axis=1,
                         pct=True) * 
                self.TSRank(self.Data['Volume']/
                            self.Data['Volume'].rolling(20,
                                 axis=0).mean(), 5).rank(axis=1, pct=True))
    
    @Output()
    def Factor_GTJA_143(self):
        temp = self.Data['Close'].pct_change(axis=0) + 1
        temp[temp < 1] = 1
        return temp.rolling(21, axis=0).apply(lambda x: x.prod())
    
    @Output()
    def Factor_GTJA_144(self):
        temp = self.Data['Close'].pct_change(axis=0)
        temp[temp > 0] = 0
        res1 = temp.abs().rolling(20, axis=0).sum()
        res2 = (temp > 0).rolling(20, axis=0).sum()
        return res1 / (1 + res2)
    
    @Output()
    def Factor_GTJA_145(self):
        return ((self.Data['Volume'].rolling(9, axis=0).mean() -
                 self.Data['Volume'].rolling(26, axis=0).mean()) /
                    self.Data['Volume'].rolling(12, axis=0).mean())
    
    @Output()
    def Factor_GTJA_146(self):
        temp = self.Data['Close'].pct_change(axis=0)
        s1 = temp - temp.ewm(alpha=1/31, axis=0).mean()
        return (s1.rolling(20, axis=0).mean() * s1 /
                ((temp - s1)**2).rolling(60, axis=0).mean())
    
    @Output()
    def Factor_GTJA_147(self):
        return self.RegBeta(self.Data['Close'].rolling(12, axis=0).mean(),
                            np.arange(12)+1)
    
    @Output()
    def Factor_GTJA_148(self):
        r1 = self.TSCorr(self.Data['Open'],
                         self.Data['Volume'].rolling(60,
                                  axis=0).mean().rolling(9, axis=0).sum(),
                                  6).rank(axis=1, pct=True)
        r2 = (self.Data['Open'] -
              self.Data['Open'].rolling(14,
                       axis=0).min()).rank(axis=1, pct=True)
        return r1 - r2
    
    @Output()
    def Factor_GTJA_149(self):
        ret = self.Data['Close'].pct_change(axis=0)
        ret_i = self.Data['Index'].pct_change(axis=0)
        ret[ret_i > 0] = np.nan
        return self.RegBeta(ret, ret_i)
    
    @Output()
    def Factor_GTJA_150(self):
        return (self.Data['Close'] + 
                self.Data['High'] + 
                self.Data['Low'])*self.Data['Volume']
        
    @Output()
    def Factor_GTJA_151(self):
        return self.Data['Close'].diff(20, 
                        axis=0).ewm(alpha=0.05, axis=0).mean()
        
    @Output()
    def Factor_GTJA_152(self):
        temp = (self.Data['Close'] /
                self.Data['Close'].shift(9,
                         axis=0)).shift(axis=0).ewm(alpha=1/9,
                         axis=0).mean().shift(axis=0)
        return (temp.rolling(12, axis=0).mean() -
                temp.rolling(26, axis=0).mean()).ewm(alpha=1/9, axis=0).mean()
        
    @Output()
    def Factor_GTJA_153(self):
        return (self.Data['Close'].rolling(3, axis=0).mean() +
                self.Data['Close'].rolling(6, axis=0).mean() +
                self.Data['Close'].rolling(12, axis=0).mean() +
                self.Data['Close'].rolling(24, axis=0).mean())/4
    
    @Output()
    def Factor_GTJA_154(self):
        return (self.TSCorr(self.Data['VWAP'], 
                            self.Data['Volume'].rolling(180,
                                     axis=0).mean(), 18) /
            (self.Data['VWAP'] - self.Data['VWAP'].rolling(16, axis=0).min()))
        
    @Output()
    def Factor_GTJA_155(self):
        temp = (self.Data['Volume'].ewm(alpha=2/13, axis=0).mean() -
                self.Data['Volume'].ewm(alpha=2/27, axis=0).mean())
        return temp - temp.ewm(alpha=0.5, axis=0).mean()
    
    @Output()
    def Factor_GTJA_156(self):
        r1 = self.Data['VWAP'].diff(5, axis=0).rolling(3,
                      axis=0).apply(lambda x: 
                          self.DecayLinear(x)).rank(axis=1, pct=True)
        r2 = (-(self.Data['Open']*0.15 + 
                self.Data['Low']*0.85).diff(2, axis=0) / 
            (self.Data['Open']*0.15 +
             self.Data['Low']*0.85)).rolling(3, axis=0).apply(lambda x:
                self.DecayLinear(x)).rank(axis=1, pct=True)
        return -np.maximum(r1, r2)
    
    @Output()
    def Factor_GTJA_157(self):
        p1 = (1 - np.log((2 - self.Data['Close'].diff(5,
                          axis=0).rank(axis=1, pct=True)).rolling(2, 
            axis=0).min()).rank(axis=1, pct=True)).rolling(5, axis=0).min()
        p2 = self.TSRank(-self.Data['Close'].pct_change(axis=0).shift(6,
                         axis=0), 5)
        return p1 + p2
    
    @Output()
    def Factor_GTJA_158(self):
        temp = self.Data['Close'].ewm(alpha=2/15, axis=0).mean()
        return (self.Data['High']  - self.Data['Low'] -
                2*temp)/self.Data['Close'] 
        
    @Output()
    def Factor_GTJA_159(self):
        p1 = ((self.Data['Close'] -
               self.Data['Low'].rolling(6, axis=0).sum())/
            (self.Data['Close'].shift(axis=0) -
             self.Data['Low']).rolling(6, axis=0).sum()*12*24)
        p2 = ((self.Data['Close'] -
               self.Data['Low'].rolling(12, axis=0).sum())/
            (self.Data['Close'].shift(axis=0) -
             self.Data['Low']).rolling(12, axis=0).sum()*6*24)
        p3 = ((self.Data['Close'] -
               self.Data['Low'].rolling(24, axis=0).sum())/
            (self.Data['Close'].shift(axis=0) -
             self.Data['Low']).rolling(24, axis=0).sum()*12*24)
        return (p1 + p2 + p3)/(100/(6*12 + 12*24 + 6*24))
    
    @Output()
    def Factor_GTJA_160(self):
        temp = self.Data['Close'].rolling(20, axis=0).std()
        temp[self.Data['Close'].diff(axis=0) > 0] = 0
        return temp.ewm(alpha=0.05, axis=0).mean()
    
    @Output()
    def Factor_GTJA_161(self):
        return np.maximum(np.maximum(self.Data['High'] - self.Data['Low'], 
                                     (self.Data['Close'].shift(axis=0) - 
                                      self.Data['High']).abs()),
            (self.Data['Close'].shift(axis=0) - 
             self.Data['High']).abs()).rolling(12, axis=0).mean()
    
    @Output()
    def Factor_GTJA_162(self):
        diff = self.Data['Close'].diff(axis=0)
        temp = (diff.ewm(alpha=1/12, axis=0).mean() /
                diff.abs().ewm(alpha=1/12, axis=0).mean())
        t_min = temp.rolling(12, axis=0).min()
        return (temp - t_min)/(temp.rolling(12, axis=0).max() - t_min)
    
    @Output()
    def Factor_GTJA_163(self):
        return (-self.Data['Close'].pct_change(axis=0) * 
                self.Data['Volume'].rolling(20, axis=0).mean() * 
                self.Data['VWAP']*(self.Data['High'] -
                         self.Data['Close'])).rank(axis=1, pct=True)
    
    @Output()
    def Factor_GTJA_164(self):
        temp = 1/self.Data['Close'].diff(axis=0)
        temp[temp < 0] = 1
        return (temp - temp.rolling(12, axis=0).min()/
                (self.Data['High'] -
                 self.Data['Low'])*100).ewm(alpha=2/13, axis=0).mean()
    
    @Output()
    def Factor_GTJA_165(self):
        pass
    
    @Output()
    def Factor_GTJA_166(self):
        ret = self.Data['Close'].pct_change(axis=0)
        return ((ret -
                 ret.rolling(20, axis=0).mean()).rolling(20,
                            axis=0).sum()/
            ((self.Data['Close']/
              self.Data['Close'].shift(axis=0)).rolling(20, 
                       axis=0).sum()**2)).rolling(20, axis=0).sum()
    
    @Output()
    def Factor_GTJA_167(self):
        res = self.Data['Close'].diff(axis=0)
        res[res > 0] = 0
        return res.rolling(12, axis=0).sum()
    
    @Output()
    def Factor_GTJA_168(self):
        return (-self.Data['Volume'] / 
                self.Data['Volume'].rolling(20, axis=0).mean())
    
    @Output()
    def Factor_GTJA_169(self):
        temp = self.Data['Close'].diff(axis=0).ewm(alpha=1/9,
                        axis=0).mean().shift(axis=0)
        return (temp.rolling(12, axis=0).mean() - 
                temp.rolling(26, axis=0).mean()).ewm(alpha=0.1, axis=0).mean()
    
    @Output()
    def Factor_GTJA_170(self):
        return ((1 - self.Data['Close'].rank(axis=1, pct=True)) *
                self.Data['Volume'] /
                self.Data['Volume'].rolling(20, axis=0).mean() *
                (self.Data['High'] *
                 (self.Data['High'] -
                  self.Data['Low']).rank(axis=1, pct=True)) /
                 (self.Data['High'].rolling(5, axis=0).mean()) -
                 self.Data['VWAP'].diff(5, axis=0).rank(axis=1, pct=True))
    
    @Output()
    def Factor_GTJA_171(self):
        return ((self.Data['Close'] - self.Data['Low']) *
                (self.Data['Open'] ** 5) / 
                (self.Data['Close'] - self.Data['High']) /
                (self.Data['Close'] ** 5))
    
    @Output()
    def Factor_GTJA_172(self):
        HD = self.Data['High'].diff(axis=0)
        LD = -self.Data['Low'].diff(axis=0)
        LD[~((LD > 0) & (LD > HD))] = 0
        HD[~((HD > 0) & (HD > LD))] = 0
        temp1 = self.Data['High'] - self.Data['Low']
        temp2 = (self.Data['High'] - self.Data['Close'].shift(axis=0)).abs()
        t_mask = temp1 < temp2
        temp1[t_mask] = temp2
        temp2 = (self.Data['Low'] - self.Data['Close'].shift(axis=0)).abs()
        t_mask = temp1 < temp2
        temp1[t_mask] = temp2
        s_TR = temp1.rolling(14, axis=0).sum()
        p1 = LD.rolling(14, axis=0).sum() / s_TR
        p2 = HD.rolling(14, axis=0).sum() / s_TR
        return ((p1 - p2).abs() / (p1 + p2)).rolling(6, axis=0).mean()
    
    @Output()
    def Factor_GTJA_173(self):
        s_c = self.Data['Close'].ewm(alpha=2/13, axis=0).mean()
        s_c_w = s_c.ewm(alpha=2/13, axis=0).mean()
        return 3*s_c - 2*s_c_w + s_c_w.ewm(alpha=2/13, axis=0).mean()
    
    @Output()
    def Factor_GTJA_174(self):
        res = self.Data['Close'].rolling(20, axis=0).std()
        res[self.Data['Close'].diff(axis=0) < 0] = 0
        return res.ewm(alpha=0.05, axis=0).mean()
    
    @Output()
    def Factor_GTJA_175(self):
        return (np.maximum(np.maximum(self.Data['High'] - self.Data['Low'],
                                      (self.Data['Close'].shift(axis=0) - 
                                       self.Data['High']).abs()), 
                (self.Data['Close'].shift(axis=0) -
                 self.Data['Low']).abs()).rolling(6, axis=0).mean())
    
    @Output()
    def Factor_GTJA_176(self):
        l_min = self.Data['Low'].rolling(12, axis=0).min()
        return self.TSCorr(((self.Data['Close'] - l_min)/
                            (self.Data['Low'].rolling(12, axis=0).max() -
                             l_min)).rank(axis=1, pct=True),
            self.Data['Volume'].rank(axis=1, pct=True), 6)
    
    @Output()
    def Factor_GTJA_177(self):
        return (20 -
                self.Data['High'].rolling(20, axis=0).apply(self.HighDay))/20
    
    @Output()
    def Factor_GTJA_178(self):
        return self.Data['Close'].pct_change(axis=0) * self.Data['Volume']
    
    @Output()
    def Factor_GTJA_179(self):
        return (self.TSCorr(self.Data['VWAP'],
                            self.Data['Volume'], 
                            4).rank(axis=1, pct=True) *
                        self.TSCorr(self.Data['Low'].rank(axis=1, pct=True),
                        self.Data['Volume'].rolling(50, 
                                 axis=0).mean().rank(axis=1, pct=True),
                                 12).rank(axis=1, pct=True))
    
    @Output()
    def Factor_GTJA_180(self):
        res = -self.Data['Volume']
        res[self.Data['Volume'].rolling(20, axis=0).mean() <
            self.Data['Volume']] = -self.TSRank(self.Data['Close'].diff(7,
            axis=0).abs(), 60) * (2 *
            ((self.Data['Close'].diff(7, axis=0) > 0) - 0.5))
    
    @Output()
    def Factor_GTJA_181(self):
        ret = self.Data['Close'].pct_change(axis=0)
        return ret - ret.rolling(20, axis=0).mean()
    
    @Output()
    def Factor_GTJA_182(self):
        d1 = self.Data['Close'] - self.Data['Open']
        d2 = self.Data['Index'] - self.Data['IndexOpen']
        return (((d1 > 0).mul(d2 > 0, axis=0)) |
                ((d1 < 0).mul(d2 < 0, axis=0))).rolling(20, axis=0).mean()
        
    @Output()
    def Factor_GTJA_183(self):
        pass
    
    @Output()
    def Factor_GTJA_184(self):
        temp = self.Data['Open'] - self.Data['Close']
        r1 = self.TSCorr(temp.shift(axis=0),
                         self.Data['Close'], 200).rank(axis=1, pct=True)
        r2 = temp.rank(axis=1, pct=True)
        return r1 + r2
    
    @Output()
    def Factor_GTJA_185(self):
        return 1 -((1 - self.Data['Open'] / 
                    self.Data['Close'])**2).rank(axis=1, pct=True)
        
    @Output()
    def Factor_GTJA_186(self):
        HD = self.Data['High'].diff(axis=0)
        LD = -self.Data['Low'].diff(axis=0)
        LD[~((LD > 0) & (LD > HD))] = 0
        HD[~((HD > 0) & (HD > LD))] = 0
        temp1 = self.Data['High'] - self.Data['Low']
        temp2 = (self.Data['High'] - self.Data['Close'].shift(axis=0)).abs()
        t_mask = temp1 < temp2
        temp1[t_mask] = temp2
        temp2 = (self.Data['Low'] - self.Data['Close'].shift(axis=0)).abs()
        t_mask = temp1 < temp2
        temp1[t_mask] = temp2
        s_TR = temp1.rolling(14, axis=0).sum()
        p1 = LD.rolling(14, axis=0).sum() / s_TR
        p2 = HD.rolling(14, axis=0).sum() / s_TR
        res = ((p1 - p2).abs() / (p1 + p2)).rolling(6, axis=0).mean()
        return res + res.shift(6, axis=0)
    
    @Output()
    def Factor_GTJA_187(self):
        res = np.maximum(self.Data['High'] - self.Data['Open'],
                         self.Data['Open'].diff(axis=0))
        res[self.Data['Open'].diff(axis=0) <= 0] = 0
        return res.rolling(20, axis=0).sum()
    
    @Output()
    def Factor_GTJA_188(self):
        temp = self.Data['High'] - self.Data['Low']
        s_t = temp.ewm(alpha=2/11, axis=0).mean()
        return temp / s_t - 1
    
    @Output()
    def Factor_GTJA_189(self):
        return self.Data['Close'].diff(6, 
                        axis=0).abs().rolling(6, axis=0).mean()
        
    @Output()
    def Factor_GTJA_190(self):
        ret = self.Data['Close'].pct_change(axis=0)
        temp1 = (self.Data['Close']/
                 self.Data['Close'].shift(19, axis=0))**0.05 - 1
        p11 = (ret > temp1).rolling(20, axis=0).sum()
        p12 = (ret < temp1).rolling(20, axis=0).sum()
        diff = ret - temp1
        p2 = diff.copy()
        p2[p2 > 0] = 0
        p2 = p2.rolling(20, axis=0).sum()
        p3 = diff.copy()
        p3[p3 < 0] = 0
        p3 = p3.rolling(20, axis=0).sum()
        return p11 * p2 / p12 / p3
    
    @Output()
    def Factor_GTJA_191(self):
        return self.TSCorr(self.Data['Volume'].rolling(20, axis=0).mean(),
                           self.Data['Low'], 5) + 0.5*(self.Data['High'] + 
                                    self.Data['Low']) - self.Data['Close']
                