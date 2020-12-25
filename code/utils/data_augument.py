# 数据扩增
import numpy as np
# --抖动,添加噪声超参数：sigma =噪声的标准偏差（STD）
def add_jitter(X, sigma=0.1):  
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    # plt.plot(myNoise, label='noise')
    return X+myNoise

# --缩放¶超参数:σ=放大/缩小系数的标准值通过乘以一个随机标量来更改窗口中数据的大小
def add_scaling(X, sigma=0.2):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1))
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

# 降采样,使用一组降采样因子 k1, k2, k3，每隔 ki-1 个数据取一个。
def down_sampling(data, rate=1):
    # down sampling by rate k
    if rate > data.shape[0] / 3:
        print('sampling rate is too high')
        return None
    ds_data = data[::rate]  # temp after down sampling
    ds_data_len = ds_data.shape[0]  # remark the length info
    return ds_data 

# --滑动平均 使用一组滑动窗口l1, l2, l3，每li个数据取平均
def moving_average(data, moving_wl=10):
    data_len = data.shape[0]
    if  moving_wl > data.shape[0] / 3:
        print('moving window is too high')
        return None
    ma_data = np.zeros(data_len-moving_wl+1)
    for i in range(data_len-moving_wl+1):
        ma_data[i] = np.mean(data[i: i+moving_wl])
    return ma_data

# ------------------裁剪（Crop） 使用滑动窗口在时间序列上截取数据
def data_crop(data, wl_ratio=0.8):
    data_len = data.shape[0]
    wl = int(data_len*wl_ratio)
    start = int(data_len*(1-wl_ratio)//2)
    end = start + wl
    #print(start, end)
    crop_data = data[start:end]
    return crop_data