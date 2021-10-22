#通常嵌入维数m的选取是2，3；
#相似度阈值r的选取是原始序列标准差的0.2倍左右
import numpy as np

def apen(L, m, r):
    """
        Calculates approximate entropy_T (ApEn) of a time series.

        Input
            L: Time series
            m: Template length
            r: Tolerance level

        Output:
            ApEn
    """
    L=np.array(L)
    N=len(L)

    # Divide time series and save all templates of length m
    xi = np.array([L[i:i + m] for i in range(N - m + 1)])

    # Compute each B_i
    B = np.array([np.sum(np.abs(xii - xi).max(axis=1) <= r) for xii in xi])/(N-m+1)
    B=np.sum(np.log(B))/(N-m+1)


    # Similar method to compute each A_i
    m += 1
    xj = np.array([L[i:i + m] for i in range(N - m + 1)])
    A = np.array([np.sum(np.abs(xjj - xj).max(axis=1) <= r) for xjj in xj])/(N-m+1)
    A = np.sum(np.log(A)) / (N - m + 1)

    # Compute and return ApEn
    ApEn = np.abs(A-B)
    return ApEn


# U = np.array([85, 80, 89] * 17)
# print (ApEn(U, 2, 3))
