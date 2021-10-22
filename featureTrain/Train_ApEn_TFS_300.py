import numpy as np
import pandas as pd
from toolbox import get_signal
from scipy.signal import stft
from ApEn import apen
from nolds import sampen
import matplotlib.pyplot as plt




'''
TPEHG数据库中
record ID     信号个数
1650         31305          非早产
873          15060          非早产
其他样本，信号个数均大于35000

这个代码主要是:
EHG时序信号经过Short Time Fourier Transform处理，被转化到时频域
在EHG信号的频率分量上计算[0,0.1,0.2,.....,3.9,4.0]近似熵+近似熵的改进
'''


# 一次STFT的窗口大小
windowSize = 200
#Hz, EHG信号原始的采样频率
sampleFrequency = 20
#最小频率
minFrequency = 0
#最大频率
maxFrequency = 4
#频率范围
frequencyRange = maxFrequency - minFrequency
#EHG信号产生的主成分个数
N_components = int(frequencyRange / (sampleFrequency / windowSize) + 1)
#EHG信号产生的特征个数，EHG信号的每个主成分可以生成近似熵特征及其改进
N_features = N_components * 2


def ComponentApEn(signal,recTime,m,x):
    """
        ComponentApEn(signal, recTime, m, x)
            Return a new array of given shape and type.
            Parameters
            ----------
            signal : EHG信号的数组，形状为[n,]
            recTime : data-type,
                EHG信号的记录时间
            m : 熵的计算参数m
            x : 熵的计算参数tolerance = x * 标准差

            Returns
            -------
            out : ndarray
                计算好的特征，近似熵及其改进

        """

    X = np.zeros(N_features)
    #reshape the signal
    signal = signal.reshape(-1,)
    # the length of the signal
    n = len(signal)
    # rescale Entropy的公式
    coefficient = recTime / 37 + 1
    f_1, t_1, Zxx_1 = stft(signal, fs=sampleFrequency, nperseg=windowSize)
    abs_Zxx_1 = np.array(np.abs(Zxx_1))
    for j in range(N_components):
        abs_Zxx_1_f = abs_Zxx_1[j, :]
        abs_Zxx_1_f_std = np.std(abs_Zxx_1_f)
        # 计算EHG信号频率成分的近似熵
        ApEn_STFT = apen(abs_Zxx_1_f, m=m, r=x * abs_Zxx_1_f_std)
        # 计算EHG信号频率成分近似熵的改进
        ApEn_STFT_coef = ApEn_STFT * coefficient
        X[0 + 2 * j] = ApEn_STFT
        X[0 + 2 * j + 1] = ApEn_STFT_coef

    return X

# signal = get_signal('F:/Entropy_STFT/data/', 'tpehg1007', [10]).reshape(-1,)
# # print(ComponentApEn(signal,31.3,3,0.125))