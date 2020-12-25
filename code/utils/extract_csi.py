# 从.dat文件中提取csi
from struct import unpack, pack
from math import sqrt 
import numpy as np


def dbinv(x):
    return 10**(x / 10)
# 计算接收到的信号强度(RSS)(以dBm为单位)
def total_rss(data):
    rssi_mag = 0
    if data['rssi_a'] != 0:
        rssi_mag = rssi_mag + dbinv(data['rssi_a'])
    if data['rssi_b'] != 0:
        rssi_mag = rssi_mag + dbinv(data['rssi_b'])
    if data['rssi_c'] != 0:
        rssi_mag = rssi_mag + dbinv(data['rssi_c'])
    return 10 * np.log10(rssi_mag) - 44 - data['agc']
# 将CSI结构转换为信道矩阵H。
def get_scaled_csi(data):
    csi = data['csi']
    ntx = data['ntx']
    nrx = data['nrx']
    csi_sq = csi * np.conj(csi)
    csi_pwr = csi_sq.sum().real  # 求和
    rssi_pwr = dbinv(total_rss(data))
    scale = rssi_pwr / (csi_pwr / 30)
    if data['noise'] == -127:
        noise = -92
    else:
        noise = data['noise']
    thermal_noise_pwr = dbinv(noise)
    quant_error_pwr = scale * (nrx * ntx)
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr
    ret = csi * sqrt(scale / total_noise_pwr)
    if ntx == 2:
        ret = ret * sqrt(2)
    elif ntx == 3:
        ret = ret * sqrt(dbinv(4.5))
    return ret

def expandable_or(x, y):
    z = x | y
    low = z & 0xff
    return unpack('b', pack('B', low))[0]

def read_bfree(array):
    result = {}
    timestamp_low = array[0] + (array[1] << 8) + (array[2] << 16) + (array[3] << 24)
    bf_count = array[4] + (array[5] << 8)
    nrx = array[8]  # 接收天线的数目
    ntx = array[9]
    rssi_a = array[10]
    rssi_b = array[11]
    rssi_c = array[12]
    # noise
    noise = unpack('b', pack('B', array[13]))[0]
    agc = array[14]
    antenna_sel = array[15]
    length = array[16] + (array[17] << 8)
    fake_rate_n_flags = array[18] + (array[19] << 8)
    calc_len = (30 * (nrx * ntx * 8 * 2 + 3) + 7) // 8
    payload = array[20:]  # csi数据部分

    if length != calc_len:
        print('数据发现错误!')
        exit(0)
    result['timestamp_low'] = timestamp_low
    result['bfree_count'] = bf_count
    result['rssi_a'] = rssi_a
    result['rssi_b'] = rssi_b
    result['rssi_c'] = rssi_c
    result['nrx'] = nrx
    result['ntx'] = ntx
    result['agc'] = agc
    result['rate'] = fake_rate_n_flags
    result['noise'] = noise
    csi = np.zeros((ntx, nrx, 30), dtype=np.complex64)
    # 现在开始构建numpy array
    idx = 0        
    for sub_idx in range(30):
        idx = idx + 3
        remainder = idx % 8  # 余数
        for m in range(nrx):
            for n in range(ntx):
                real = expandable_or((payload[idx // 8] >> remainder), (payload[idx // 8 + 1] << (8 - remainder)))
                img = expandable_or((payload[idx // 8 + 1] >> remainder), (payload[idx // 8 + 2] << (8 - remainder)))
                csi[n, m, sub_idx] = complex(real, img)     # 构建一个复数
                idx = idx + 16
    result['csi'] = csi
    perm = np.zeros(3, dtype=np.uint32)
    perm[0] = (antenna_sel & 0x3) + 1
    perm[1] = ((antenna_sel >> 2) & 0x3) + 1
    perm[2] = ((antenna_sel >> 4) & 0x3) + 1
    result['perm'] = perm
    return result

# 从.dat抽取生成CSI字典数组 2*3*30
def extract_csi(file_name):
    triangle = np.array([1, 3, 6])
    csis = []
    with open(file_name, 'rb') as f:
        buff = f.read()
        curr = 0    # 记录当前已经处理到了的位置
        length = len(buff)
        while curr < (length - 3):
            data_len = unpack('>H', buff[curr:curr+2])[0]  # 实际长度
            if data_len > (length - curr - 2):  # 防止越界的错误
                break
            code = unpack('B', buff[curr+2:curr+3])[0]  # 代码
            curr = curr + 3
            if code == 187:
                # 将CSI数据帧解析
                csi_dic = read_bfree(buff[curr:])
                perm = csi_dic['perm']
                nrx = csi_dic['nrx']
                csi = csi_dic['csi']
                if sum(perm) == triangle[nrx - 1]:  # 下标从0开始
                    csi[:, perm - 1, :] = csi[:, [x for x in range(nrx)], :]
                # csi = get_scaled_csi(data)
                csis.append(csi_dic)
            curr = curr + data_len - 1
    return csis