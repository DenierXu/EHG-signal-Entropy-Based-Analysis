import numpy as np
from toolbox import get_signal
from scipy.signal import stft
from nolds import sampen




def ComponentSampEn(signal, recTime, m, x, windowSize=200, sampleFrequency=20, minFrequency=0, maxFrequency=4):
    """
        ComponentSampEn(signal, recTime, m, x,, windowSize=200, sampleFrequency=20, minFrequency=0, maxFrequency=4)
            Return a new array of given shape and type.
            Parameters
            ----------
            signal : EHG信号的数组，形状为[n,]
            recTime : data-type,
                EHG信号的记录时间
            m : 熵的计算参数m
            x : 熵的计算参数tolerance = x * 标准差
            windowSize: STFT的窗口大小
            sampleFrequency:EHG信号原始的采样频率
            minFrequency:最小频率,必须为0，否则需要改写代码
            maxFrequency:最大频率

            Returns
            -------
            out : ndarray
                计算好的特征，样本熵及其改进

        """

    # 频率范围
    frequencyRange = maxFrequency - minFrequency
    # EHG信号产生的主成分个数
    N_components = int(frequencyRange / (sampleFrequency / windowSize) + 1)
    # EHG信号产生的特征个数，EHG信号的每个主成分可以生成样本熵特征及其改进
    N_features = N_components * 2
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
        # 计算EHG信号频率成分的样本熵
        SampEn_STFT = sampen(abs_Zxx_1_f, emb_dim=m, tolerance=x * abs_Zxx_1_f_std)
        # 计算EHG信号频率成分样本熵的改进
        SampEn_STFT_coef = SampEn_STFT * coefficient
        X[0 + 2 * j] = SampEn_STFT
        X[0 + 2 * j + 1] = SampEn_STFT_coef

    return X
