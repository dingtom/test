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
import random
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from utils.extract_csi import extract_csi, get_scaled_csi
from utils.get_files_info import get_label_data, get_file_list, get_crnn_cross_train_data, get_crnn_cross_batch_generator
from utils.models import create_crnn_cross_model

# 设置GPU内存按需分配
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default='/home/zut_csi/tomding/RCNN/data_processed', type=str, help="输入数据dir")  # required=True,
parser.add_argument("--image_size", default=64, type=int, help="")
parser.add_argument("--logs_path", default='./logs', type=str, help="模型存储在何处")
parser.add_argument("--test_mode", action='store_true', help="Whether to run training.")
parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
parser.add_argument("--batch_size", default=64, type=int, help="Total batch size for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--epochs", default=1000, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
args = parser.parse_args()

if not os.path.exists(args.logs_path):  # 判断保存模型的目录是否存在
    os.makedirs(args.logs_path)  # 如果不存在，就新建一个

label_file_list, wav_file_list = get_file_list(args.data_path)
if args.test_mode == True:
    label_file_list, wav_file_list = label_file_list[:1000], wav_file_list[:1000]
    print('_________________________________________runing in test_mode_________________________________________')
train_label_file_list, test_label_file_list, train_wav_file_list, test_wav_file_list = train_test_split(label_file_list, wav_file_list, test_size=0.1, random_state=123)
# 用迭代器的时候需要
train_wav_file_list, validate_wav_file_list, train_label_file_list, validate_label_file_list = train_test_split(train_wav_file_list, train_label_file_list, test_size=0.1, random_state=123)
validate_label_list = get_label_data(train_label_file_list)

train_label_list = get_label_data(train_label_file_list)
test_label_list = get_label_data(test_label_file_list)

cb = []
cb.append(keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=50, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0))
# 当监测值不再改善时，该回调函数将中止训练可防止过拟合
cb.append(keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, verbose=1, mode='auto'))
crnn_cross_model = create_crnn_cross_model(input_size=(args.image_size, args.image_size, 1), output_size=4) 
print(crnn_cross_model.summary())
print('train files amount:', len(train_label_file_list), len(train_wav_file_list))#, 'validate files amount:', len(validate_label_list), validate_file_nums,)
print('test files amount:', len(test_label_file_list), len(test_wav_file_list))
# keras.utils.plot_model(base_model, show_shapes=True)

# 用迭代器的时候需要
train_batch_gen = get_crnn_cross_batch_generator(args.batch_size, train_wav_file_list, train_label_list, args.image_size)
validate_batch_gen = get_crnn_cross_batch_generator(args.batch_size, validate_wav_file_list, validate_label_list, args.image_size)
test_batch_gen = get_crnn_cross_batch_generator(args.batch_size, test_wav_file_list, test_label_list, args.image_size)

# input_data = next(train_batch_gen)[0]
# plt.imshow(input_data['the_inputs'][0].T[0])
# plt.show()
# print(input_data['the_inputs'].shape, input_data['the_labels'].shape, input_data['input_length'].shape, input_data['label_length'].shape)
his = crnn_cross_model.fit_generator(train_batch_gen, verbose=1, steps_per_epoch=len(train_wav_file_list)//args.batch_size, validation_data=validate_batch_gen, validation_steps=len(validate_wav_file_list)//args.batch_size, epochs=args.epochs, callbacks=cb)  
test_wavs, test_labels = get_crnn_cross_train_data(test_wav_file_list, test_label_list, args.image_size)

acc = crnn_cross_model.evaluate(test_wavs, test_labels)
print(acc)
# # 读入所有数据
# #train_data = get_crnn_cross_train_data(train_wav_file_list, train_label_list, args.image_size)
# #  print(train_data[0]['the_inputs'].shape, train_data[1]['the_labels'].shape, 2222222222222222222)
# # # his = crnn_cross_model.fit(train_data[0], train_data[1], validation_split=0.1, batch_size=args.batch_size, epochs=args.epochs, callbacks=cb)  # callback的epoch都是对fit里的参数来说

# #  保存模型结构及权重
# crnn_cross_model.save_weights('./train_files/crnn_cross_save_weights.h5')
# with open('./train_files/crnn_cross_model_struct.json', 'w') as f:
#     json_string = crnn_cross_model.to_json()
#     f.write(json_string)  # 保存模型信息
# print('模型结构及权重已保存')

# # 加载权重
# with open('./train_files/crnn_cross_model_struct.json') as f:
#     model_struct = f.read()
# test_model = keras.models.model_from_json(model_struct)
# test_model.load_weights('./train_files/crnn_cross_save_weights.h5')
# # model = keras.models.load_model('all_model.h5')
# print('模型已加载')


# # 对模型进行评价
# def evaluate(kind, wavs, labels):
#     wavs, labels = wavs['the_inputs'], labels['the_labels']
#     data_num = len(wavs)
#     error_cnt = 0
#     for i in range(data_num):
#         pre_label = int(test_model.predict(wavs[i][np.newaxis, :]) ) # (1, 20, 11)
#         label = int(labels[i])
#         if label != pre_label:
#             error_cnt += 1
#             print('真实标签：', label, '预测结果', pre_label)
#     print('{}:样本数{}错误数{}准确率：{:%}'.format(kind, data_num, error_cnt, (1-error_cnt/data_num)))


# # 训练集
# train_wavs, train_labels = get_crnn_cross_train_data(train_wav_file_list, train_label_list, args.image_size)
# # 测试集
# test_wavs, test_labels = get_crnn_cross_train_data(test_wav_file_list, test_label_list, args.image_size)
# # print(train_wavs['the_inputs'].shape, train_labels['the_labels'].shape)
# evaluate('trian', train_wavs, train_labels)
# evaluate('test', test_wavs, test_labels)
