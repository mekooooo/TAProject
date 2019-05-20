# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:21:23 2019

@author: Brian

"""
import sys
sys.path.append('../technicalFactors/')
from TechnicalAnalysis import TechnicalAnalysis

if __name__ == '__main__':
    
    data_dir = ''
    factor_dir = '{}'.format(data_dir)
    TA = TechnicalAnalysis(output=factor_dir)
    
#    TA.Factor_STO_TRIX.__code__.co_varnames[1:TA.Factor_STO_TRIX.__code__.co_argcount]
    temp = TA.Factor_ADX(window = 5, rolling = 5)
    temp = TA.Factor_ADX(window = 5, rolling = 10)
    temp = TA.Factor_ADX(window = 5, rolling = 22)
    temp = TA.Factor_ADX(window = 10, rolling = 5)
    temp = TA.Factor_ADX(window = 10, rolling = 10)
    temp = TA.Factor_ADX(window = 10, rolling = 22)
    temp = TA.Factor_BBands(window = 10)
    temp = TA.Factor_BBands(window = 22)
    temp = TA.Factor_BBandsD(short = 5, long = 10)
    temp = TA.Factor_BBandsD(short = 5, long = 22)
    temp = TA.Factor_BBandsD(short = 5, long = 66)
    temp = TA.Factor_BBandsD(short = 10, long = 22)
    temp = TA.Factor_BBandsD(short = 10, long = 66)
    temp = TA.Factor_BBandsD(short = 22, long = 66)
    temp = TA.Factor_CProb(short = 3, long = 5, lookback = 21)
    temp = TA.Factor_CProb(short = 5, long = 10, lookback = 21)
    temp = TA.Factor_CProb(short = 5, long = 10, lookback = 66)
    temp = TA.Factor_CProb(short = 10, long = 21, lookback = 66)
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
#    temp = TA.Factor_MADMOM(lookback = 10, short = 3, long = 22)
#    temp = TA.Factor_MADMOM(lookback = 10, short = 10, long = 22)
#    temp = TA.Factor_MADMOM(lookback = 10, short = 3, long = 200)
#    temp = TA.Factor_MADMOM(lookback = 10, short = 10, long = 200)
#    temp = TA.Factor_MADMOM(lookback = 10, short = 22, long = 200)
#    temp = TA.Factor_MADMOM(lookback = 22, short = 3, long = 22)
#    temp = TA.Factor_MADMOM(lookback = 22, short = 10, long = 22)
#    temp = TA.Factor_MADMOM(lookback = 22, short = 3, long = 200)
#    temp = TA.Factor_MADMOM(lookback = 22, short = 10, long = 200)
#    temp = TA.Factor_MADMOM(lookback = 22, short = 22, long = 200)
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
#    temp = TA.Factor_PPSR(lookback = 22)
#    temp = TA.Factor_PPSR(lookback = 66)
#    temp = TA.Factor_PPSR(lookback = 132)
    temp = TA.Factor_RSI_bot(short = 3, rolling = 5)
    temp = TA.Factor_RSI_bot(short = 3, rolling = 10)
    temp = TA.Factor_RSI_bot(short = 5, rolling = 5)
    temp = TA.Factor_RSI_bot(short = 5, rolling = 10)
    temp = TA.Factor_RSI_bot(short = 10, rolling = 5)
    temp = TA.Factor_RSI_bot(short = 10, rolling = 10)
    temp = TA.Factor_RSI_comp(short = 3, long = 10, rolling = 10)
    temp = TA.Factor_RSI_comp(short = 3, long = 10, rolling = 22)
    temp = TA.Factor_RSI_comp(short = 5, long = 22, rolling = 5)
    temp = TA.Factor_RSI_comp(short = 5, long = 22, rolling = 10)
    temp = TA.Factor_RSI_comp(short = 5, long = 22, rolling = 22)
    temp = TA.Factor_RSI_comp(short = 10, long = 22, rolling = 5)
    temp = TA.Factor_RSI_comp(short = 10, long = 22, rolling = 10)
    temp = TA.Factor_RSI_comp(short = 10, long = 22, rolling = 22)
    temp = TA.Factor_RSI_low(window = 5)
    temp = TA.Factor_RSI_low(window = 10)
    temp = TA.Factor_RSI_low(window = 22)
    temp = TA.Factor_RelRev(window=10, rolling=10)
    temp = TA.Factor_RelRev(window=21, rolling=10)
    temp = TA.Factor_RelRev(window=66, rolling=10)
    temp = TA.Factor_RelRev(window=132, rolling=10)
    temp = TA.Factor_RelRev(window=10, rolling=22)
    temp = TA.Factor_RelRev(window=21, rolling=22)
    temp = TA.Factor_RelRev(window=66, rolling=22)
    temp = TA.Factor_RelRev(window=132, rolling=22)
    temp = TA.Factor_Reversal(window = 3)
    temp = TA.Factor_Reversal(window = 5)
    temp = TA.Factor_Reversal(window = 10)
    temp = TA.Factor_Reversal(window = 22)
    temp = TA.Factor_Reversal(window = 66)
    temp = TA.Factor_Reversal(window = 132)
    temp = TA.Factor_STO(window = 5, rolling = 5)
    temp = TA.Factor_STO(window = 5, rolling = 10)
    temp = TA.Factor_STO(window = 5, rolling = 22)
    temp = TA.Factor_STO(window = 10, rolling = 5)
    temp = TA.Factor_STO(window = 10, rolling = 10)
    temp = TA.Factor_STO(window = 10, rolling = 22)
    temp = TA.Factor_STO(window = 22, rolling = 5)
    temp = TA.Factor_STO(window = 22, rolling = 10)
    temp = TA.Factor_STO(window = 22, rolling = 22)
    temp = TA.Factor_STO_TRIX(window = 5)
    temp = TA.Factor_STO_TRIX(window = 10)
    temp = TA.Factor_STO_TRIX(window = 22)
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
    temp = TA.Factor_WPO(rolling=5)
    temp = TA.Factor_WPO(rolling=10)
    temp = TA.Factor_WPO(rolling=21)
    temp = TA.Factor_std(window=5)
    temp = TA.Factor_std(window=10)
    temp = TA.Factor_std(window=21)
    temp = TA.Factor_std(window=66)
    temp = TA.Factor_std2(window=5, rolling=10)
    temp = TA.Factor_std2(window=5, rolling=21)
    temp = TA.Factor_std2(window=10, rolling=10)
    temp = TA.Factor_std2(window=10, rolling=21)
    temp = TA.Factor_std2(window=21, rolling=21)
    temp = TA.Factor_std2(window=21, rolling=66)
    temp = TA.Factor_std2(window=66, rolling=21)
    temp = TA.Factor_std2(window=66, rolling=66)
    temp = TA.NightGap(rolling=3)
    temp = TA.NightGap(rolling=10)
    temp = TA.NightGap(rolling=22)


    for func in [getattr(TA, x) for x in dir(TA) if 'GTJA' in x]:
        temp = func()
    temp = TA.Factor_GTJA_116()
    temp = TA.Factor_GTJA_117()
    temp = TA.Factor_GTJA_121()
    temp = TA.Factor_GTJA_131()
    temp = TA.Factor_GTJA_138()
    temp = TA.Factor_GTJA_140()
    temp = TA.Factor_GTJA_144()
    temp = TA.Factor_GTJA_147()
    temp = TA.Factor_GTJA_149()
    temp = TA.Factor_GTJA_157()
    temp = TA.Factor_GTJA_21()
    temp = TA.Factor_GTJA_44()
    
    
    