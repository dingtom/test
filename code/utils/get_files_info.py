#　加载数据生成特征，供网络训练
import os
import random
from tqdm import tqdm
import numpy as np  
import scipy.io as scio
import scipy.signal as signal
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from pyts.image import *
from utils.data_augument import *

random.seed(123)
# 读取文件,生成音频文件和标签文件列表
def get_file_list(train_file):
    label_f_list = []
    wav_f_list = []
    for root, dirs, files in os.walk(train_file):
        for file in files:
            if file.endswith('.mat') or file.endswith('.MAT'):
                wav_file = os.sep.join([root, file])
                label_file = wav_file.split('.mat')[0] + '.txt'
                wav_f_list.append(wav_file)
                label_f_list.append(label_file)
    return label_f_list, wav_f_list

# label数据处理
def get_label_data(label_f_list):
    # 生成label_data每个文件里的声音标签集合
    l_list = []
    for label_file in tqdm(label_f_list):
        with open(label_file, 'r', encoding='utf8') as ff:
            try:
                data = ff.read()
            except:
                print(label_file, 'not get label data')
            l_list.append(data)
    return l_list

# 为label建立词典
def gen_py_list(label_batch):
    p_list = []
    for li in label_batch:
        l = li.split('  ')
        for pny in l:
            if pny not in p_list:
                p_list.append(pny)
    p_list.append('_')  # 该帧可能是空
    return p_list

# 将读取到的label映射到对应的id
def py2id(l, p_list):
    ids = []
    for py in l.split('  '):
        try:
            ids.append(p_list.index(py))
        except ValueError:
            print(py, 'is not in py_llist')
    return ids

# 对label进行padding和长度获取，不同的是数据维度不同，且label的长度就是输入给ctc的长度，不需要额外处理
def label_padding(label_batch):
    label_lens = np.array([len(l) for l in label_batch])
    label_max_len = max(label_lens)
    new_label_batch = np.zeros((len(label_batch), label_max_len))
    for j in range(len(label_batch)):
        new_label_batch[j][:len(label_batch[j])] = label_batch[j]
    return new_label_batch, label_lens

# 统一batch内数据：[batch_size, time_step, feature_dim],除此之外，ctc需要获得的信息还有输入序列的长度。
def wav_padding(image_size, wav_batch):
    wav_lens = [len(w) for w in wav_batch]
    wav_max_len = max(wav_lens)
    # 每一个sample的时间长都不一样，选择batch内最长的那个时间为基准，进行padding。
    # wav_max_len, len(wav_batch[0][0], 1)))
    new_wav_batch = np.zeros((len(wav_batch), wav_max_len, image_size, 1))
    # 需要构成成一个tensorflow块，这就要求每个样本数据形式是一样的。
    for j in range(len(wav_batch)):
        new_wav_batch[j, :wav_batch[j].shape[0], :, 0] = wav_batch[j]
    # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
    wav_length = np.array([j // 8 for j in wav_lens])
    return new_wav_batch, wav_length

# 生成batch_size的信号时频图和标签数据，存放到两个list中去
def get_crnn_ctc_batch_generator(b_size, w_list, l_list, p_list, image_size):
    shuffle_list = [i for i in range(len(w_list))]
    while True:
        for j in range(len(w_list)//b_size):
            random.shuffle(shuffle_list)  # 打乱数据的顺序，我们通过查询乱序的索引值，来确定训练数据的顺序
            wav_batch = []
            label_batch = []
            begin = j*b_size
            end = begin+b_size
            for index in shuffle_list[begin:end]:
                fbank = compute_fbank_filt(w_list[index], image_size)
                # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
                # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
                pad_fbank = np.zeros((image_size//8*8+8, image_size))
                pad_fbank[:fbank.shape[0], :] = fbank
                label = py2id(l_list[index], p_list)
                wav_batch.append(pad_fbank)
                label_batch.append(label)
            pad_wav_data, wav_length = wav_padding(image_size, wav_batch)
            pad_label_data, label_length = label_padding(label_batch)
            input_batch = {'the_inputs': pad_wav_data,
                           'the_labels': pad_label_data,
                           'input_length': wav_length,
                           'label_length': label_length}
            output_batch = {'ctc': np.zeros(pad_wav_data.shape[0])}

            yield input_batch, output_batch

# 生成batch_size的信号时频图和标签数据，存放到两个list中去
def get_crnn_cross_batch_generator(b_size, w_list, l_list, image_size):
    shuffle_list = [i for i in range(len(w_list))]
    while True:
        for j in range(len(w_list)//b_size):
            random.shuffle(shuffle_list)  # 打乱数据的顺序，我们通过查询乱序的索引值，来确定训练数据的顺序
            wav_batch = []
            label_batch = []
            begin = j*b_size
            end = begin+b_size
            for index in shuffle_list[begin:end]:
                X = np.squeeze(scio.loadmat(w_list[index])['csi'][:, 0]) # !!!!!!!!!!!!!!!!!!!!!!!!!!!子载波
                # print(type(X),X.shape)
                gasf = GADF(image_size) 
                X_gasf = gasf.fit_transform(X.reshape(1, -1)) # (1, 64, 64)
                fbank = X_gasf[0]  
                # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
                # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
                wav_batch.append(fbank)
                
                l =int(l_list[index])
                label = to_categorical(l-1, num_classes=4)
                #plt.imshow(fbank, cmap='binary', origin='lower')
                #plt.savefig(w_list[index].replace('mat', 'jpg'))
                label_batch.append(label)
            input_batch = {'the_inputs': np.array(wav_batch).reshape(-1, 64, 64, 1)}
            output_batch = {'the_labels': np.array(label_batch)}

            yield input_batch, output_batch

def get_crnn_ctc_train_data(w_list, l_list, p_list, image_size):
    wav_batch, label_batch = [], []
    for index in range(len(w_list)):
        X = np.squeeze(scio.loadmat(w_list[index])['csi'][:, 0]) # n*3 -> n# !!!!!!!!!!!!!!!!!!!!!!!!!!!子载波
        # print(type(X),X.shape)
        gasf = GADF(image_size)
        X_gasf = gasf.fit_transform(X.reshape(1, -1))
        fbank = X_gasf[0]
        # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
        # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
        wav_batch.append(fbank)
        label = py2id(l_list[index], p_list)
        label_batch.append(label)

        X1 = add_jitter(X)
        X_gasf1 = gasf.fit_transform(X1.reshape(1, -1))
        fbank1 = X_gasf1[0]
        # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
        # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
        wav_batch.append(fbank1)
        label_batch.append(label)

        X2 = add_scaling(X)
        X_gasf2 = gasf.fit_transform(X2.reshape(1, -1))
        fbank2 = X_gasf2[0]   # (1, 64, 64)
        # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
        # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
        wav_batch.append(fbank2)
        label_batch.append(label)

        X3 = down_sampling(X)
        X_gasf3 = gasf.fit_transform(X3.reshape(1, -1))
        fbank3 = X_gasf3[0]
        # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
        # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
        wav_batch.append(fbank3)
        label_batch.append(label)

        X4 = moving_average(X)
        X_gasf4 = gasf.fit_transform(X4.reshape(1, -1))
        fbank4 = X_gasf4[0]
        # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
        # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
        wav_batch.append(fbank4)
        label_batch.append(label)

        X5 = data_crop(X)
        X_gasf5 = gasf.fit_transform(X5.reshape(1, -1))
        fbank5 = X_gasf5[0]
        # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
        # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
        wav_batch.append(fbank5)
        label_batch.append(label)

    pad_wav_data, wav_length = wav_padding(image_size, wav_batch)
    pad_label_data, label_length = label_padding(label_batch)
    inputs = {'the_inputs': pad_wav_data,
              'the_labels': pad_label_data,
              'input_length': wav_length,
              'label_length': label_length}
    outputs = {'ctc': np.zeros(pad_wav_data.shape[0])}
    return inputs, outputs

def get_crnn_cross_train_data(w_list, l_list,image_size):
    wav_batch, label_batch = [], []
    for index in range(len(w_list)):
        label = int(l_list[index])

        X = np.squeeze(scio.loadmat(w_list[index])['csi'][:, 0]) # !!!!!!!!!!!!!!!!!!!!!!!!!!!子载波
        # print(type(X),X.shape)
        gasf = GADF(image_size) 
        X_gasf = gasf.fit_transform(X.reshape(1, -1)) # (1, 64, 64)
        fbank = X_gasf[0]  
        # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
        # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
        wav_batch.append(fbank)
        label_batch.append(label)
        

        X1 = add_jitter(X)
        X_gasf1 = gasf.fit_transform(X1.reshape(1, -1))
        fbank1 = X_gasf1[0]
        # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
        # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
        wav_batch.append(fbank1)
        label_batch.append(label)

        X2 = add_scaling(X)
        X_gasf2 = gasf.fit_transform(X2.reshape(1, -1))
        fbank2 = X_gasf2[0]
        # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
        # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
        wav_batch.append(fbank2)
        label_batch.append(label)

        X3 = down_sampling(X)
        X_gasf3 = gasf.fit_transform(X3.reshape(1, -1))
        fbank3 = X_gasf3[0]
        # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
        # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
        wav_batch.append(fbank3)
        label_batch.append(label)

        X4 = moving_average(X)
        X_gasf4 = gasf.fit_transform(X4.reshape(1, -1))
        fbank4 = X_gasf4[0]
        # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
        # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
        wav_batch.append(fbank4)
        label_batch.append(label)

        X5 = data_crop(X)
        X_gasf5 = gasf.fit_transform(X5.reshape(1, -1))
        fbank5 = X_gasf5[0]
        # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
        # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
        wav_batch.append(fbank5)
        label_batch.append(label)

    inputs = {'the_inputs': np.array(wav_batch).reshape(-1, 64, 64, 1)}
    outputs = {'the_labels': np.array(label_batch)}
    return inputs, outputs




 # 生成的信号时频图和标签数据，存放到两个list中去
def get_crnn_ctc_test_data(image_size, w_list, l_list, p_list):
    input_data = []
    output_data = []
    for i in range(len(w_list)):
        wav_batch = []
        label_batch = []

        X = np.squeeze(scio.loadmat(w_list[i])['csi'][:, 0]) # !!!!!!!!!!!!!!!!!!!!!!!!!!!子载波
        # print(type(X),X.shape)
        gasf = GADF(image_size)
        X_gasf = gasf.fit_transform(X.reshape(1, -1))
        fbank = X_gasf[0]
        # !!!!!!!3个maxpooling层数据的每个维度需要能够被8整除。因此我们训练实际输入的数据为wav_len//8。!!!!!!!!!!!!
        # fbank.shape[0]//8*8+8, fbank.shape[1], 3))
        wav_batch.append(fbank)
        label = py2id(l_list[i], p_list)
        label_batch.append(label[0])
        pad_wav_data, wav_length = wav_padding(image_size, wav_batch)
        input_data.append(pad_wav_data)
        output_data.append(label[0])
    return input_data, output_data