# -*- coding: utf-8 -*-
import json
import re
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
# 取值 0 ： 0也是默认值，输出所有信息
# 取值 1 ： 屏蔽通知信息
# 取值 2 ： 屏蔽通知信息和警告信息
# 取值 3 ： 屏蔽通知信息、警告信息和报错信息
import threading
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from utils.extract_csi import extract_csi, get_scaled_csi
# data_crop, down_sampling, moving_average, add_jitter, add_scaling
from utils.get_files_info import gen_py_list, get_label_data, get_file_list, get_crnn_ctc_train_data, get_crnn_ctc_test_data
from utils.models import create_crnn_ctc_model, decode_ctc
from utils.ploter import plot_confusion_matrix



import warnings
warnings.filterwarnings('ignore')
# 设置GPU内存按需分配
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default='/home/zut_csi/tomding/RCNN/data_processed/all/1', type=str, help="输入数据dir")  # required=True,
parser.add_argument("--image_size", default=64, type=int, help="")
parser.add_argument("--logs_path", default='./logs', type=str, help="模型存储在何处")
parser.add_argument("--test_mode", action='store_true', help="Whether to run training.")  # 运行时该变量有传参就将该变量设为True
parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
parser.add_argument("--batch_size", default=128, type=int, help="Total batch size for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--epochs", default=1000, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
args = parser.parse_args()

np.random.seed(args.seed)

label_file_list, wav_file_list = get_file_list(args.data_path)
if args.test_mode == True:
    label_file_list, wav_file_list = label_file_list[:1000], wav_file_list[:1000]
    print('_________________________________________runing in test_mode_________________________________________')
train_label_file_list, test_label_file_list, train_wav_file_list, test_wav_file_list = train_test_split(label_file_list, wav_file_list, test_size=0.1, random_state=42)
train_label_list = get_label_data(train_label_file_list)

# 用迭代器的时候需要
# train_wav_file_list, validate_wav_file_list, train_label_list, validate_label_list = train_test_split(train_wav_file_list, train_label_list, test_size=0.2, random_state=0)
test_label_list = get_label_data(test_label_file_list)

# 每个文件拼音标签的集合
with open('./train_files/crnn_ctc_label_list.txt', 'w') as f:  
    f.write('\n'.join(train_label_list))  
py_list = gen_py_list(train_label_list)  # 拼音的集合
with open('./train_files/crnn_ctc_pinyin_list.txt', 'w') as f:
    f.write('\n'.join(py_list))  # 保存拼音列表
print('py_list len:', len(py_list), 'py_list:', py_list)


base_model, crnn_ctc_model = create_crnn_ctc_model(input_size=(args.image_size, args.image_size, 1), output_size=len(py_list)) 
# print(crnn_ctc_model.summary())
# keras.utils.plot_model(crnn_cross_model, show_shapes=True)

if not os.path.exists(args.logs_path):  # 判断保存模型的目录是否存在
    os.makedirs(args.logs_path)  # 如果不存在，就新建一个，避免之后保存模型的时候炸掉

# train_batch_gen = get_batch_generator(batch_size, train_wav_file_list, train_label_list, py_list, image_size)
# validate_batch_gen = get_batch_generator(batch_size, validate_wav_file_list, validate_label_list, py_list, image_size)
# input_data = next(train_batch_gen)[0]
# plt.imshow(input_data['the_inputs'][0].T[0])
# plt.show()
# print(input_data['the_inputs'].shape, input_data['the_labels'].shape, input_data['input_length'].shape, input_data['label_length'].shape)

train_data = get_crnn_ctc_train_data(train_wav_file_list, train_label_list, py_list, args.image_size)

cb = []
cb.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0))
# 当监测值不再改善时，该回调函数将中止训练可防止过拟合
cb.append(keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, verbose=1, mode='auto'))
# his = ctc_model.fit_generator(train_batch_gen, verbose=1, steps_per_epoch=train_file_nums//batch_size, validation_data=validate_batch_gen, validation_steps=validate_file_nums//batch_size, epochs=1000, callbacks=cb)  
if args.test_mode == True:
    print('_________________________________________runing in test_mode_________________________________________')
print('train files amount:', len(train_label_file_list), len(train_wav_file_list))#, 'validate files amount:', len(validate_label_list), validate_file_nums,)
print('test files amount:', len(test_label_file_list), len(test_wav_file_list))

#his = crnn_ctc_model.fit(train_data[0], train_data[1], validation_split=0.1, batch_size=args.batch_size, epochs=args.epochs, callbacks=cb)  # callback的epoch都是对fit里的参数来说

#  保存模型结构及权重
crnn_ctc_model.save_weights(r'./train_files/crnn_ctc_save_weights.h5')
with open(r'./train_files/crnn_ctc_model_struct.json', 'w') as f:
    json_string = base_model.to_json()
    f.write(json_string)  # 保存模型信息
print('模型结构及权重已保存')

# 加载权重
with open(r'./train_files/crnn_ctc_model_struct.json') as f:
    model_struct = f.read()
test_model = keras.models.model_from_json(model_struct)
test_model.load_weights(r'./train_files/crnn_ctc_save_weights.h5')
# model = keras.models.load_model('all_model.h5')
print('模型已加载')
py_list = []
with open(r'./train_files/crnn_ctc_pinyin_list.txt', 'r') as f:
    contents = f.readlines()
for line in contents:
    i = line.strip('\n')
    py_list.append(i)
print('py_list已加载')


# 对模型进行评价
def evaluate(kind, wavs, labels, p):
    data_num = len(wavs)
    error_cnt = 0
    for i in range(data_num):
        pre = test_model.predict(wavs[i])  # (1, 20, 11)
        pre_index, pre_label = decode_ctc(pre, py_list)  # ['5']
        try:
            pre_label = int(pre_label[0])
        except:
            pre_label = None
        label = int(py_list[labels[i]])
        p.append(label)
        if label != pre_label:
            error_cnt += 1
            print('真实标签：', label, '预测结果', pre_label)
    print('{}:样本数{}错误数{}准确率：{:%}'.format(kind, data_num, error_cnt, (1-error_cnt/data_num)))


# 训练集
train_wavs, train_labels = get_crnn_ctc_test_data(args.image_size, train_wav_file_list, train_label_list, py_list)
# 测试集
test_wavs, test_labels = get_crnn_ctc_test_data(args.image_size, test_wav_file_list, test_label_list, py_list)
p = []
evaluate('trian', train_wavs, train_labels, p)
p = []
evaluate('test', test_wavs, test_labels, p)
print(len(test_labels), len(test_wavs))
# acc = test_model.evaluate(test_wavs, test_labels)
# print(acc, 1111111111111)
plot_confusion_matrix('Confusion Matrix', test_labels, p, py_list)
plot_confusion_matrix('Confusion Matrix', test_labels, test_model.predict_on_batch(test_wavs), py_list)