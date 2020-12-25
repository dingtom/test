import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import numpy as np

# 搭建模型
# 添加CTC损失函数
def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
# 定义解码器
def decode_ctc(preds, py_list):
    window_num = np.zeros((1), dtype=np.int32)
    window_num[0] = preds.shape[1]
    decode = keras.backend.ctc_decode(
        preds, window_num, greedy=True, beam_width=100, top_paths=1)
    result_index = keras.backend.get_value(decode[0][0])[0]
    result_py = []
    for i in result_index:
        try:
            result_py.append(py_list[i])
        except IndexError:
            print(i, 'not in py_list')
    return result_index, result_py

# ctc 损失函数的CRNN
def create_crnn_ctc_model(input_size, output_size):
    inputs = Input(name='the_inputs', shape=input_size)
    # 1
    h1_1 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_1-1')(inputs)
    h1_2 = BatchNormalization(name='BatchNormal_1-1')(h1_1)
    h1_3 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_1-2')(h1_2)
    h1_4 = BatchNormalization(name='BatchNormal_1-2')(h1_3)
    h1_5 = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", name='MaxPooling2D_1')(h1_4)
    # 2
    h2_1 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_2-1')(h1_5)
    h2_2 = BatchNormalization(name='BatchNormal_2-1')(h2_1)
    h2_3 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_2-2')(h2_2)
    h2_4 = BatchNormalization(name='BatchNormal_2-2')(h2_3)
    h2_5 = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", name='MaxPooling2D_2')(h2_4)
    # 3
    h3_1 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_3-1')(h2_5)
    h3_2 = BatchNormalization(name='BatchNormal_3-1')(h3_1)
    h3_3 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_3-2')(h3_2)
    h3_4 = BatchNormalization(name='BatchNormal_3-2')(h3_3)
    h3_5 = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", name='MaxPooling2D_3')(h3_4)
    # 4
    h4_1 = Conv2D(512, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_4-1')(h3_5)
    h4_2 = BatchNormalization(name='BatchNormal_4-1')(h4_1)
    h4_3 = Conv2D(512, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_4-2')(h4_2)
    h4_4 = BatchNormalization(name='BatchNormal_4-2')(h4_3)
    # 由于声学模型网络结构原因（3个maxpooling层），我们的音频数据的每个维度需要能够被8整除。这里输入序列经过卷积网络后，长度缩短了8倍，因此我们训练实际输入的数据为wav_len//8。
    h5_1 = Reshape((-1, int(input_size[1]//8*512)), name='Reshape_1')(h4_4)
    lstm_1 = LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(h5_1)
    lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm_1)
    h5_2 = Dense(512, activation='relu', use_bias=True, kernel_initializer='he_normal', name='Dense_1')(lstm_2)
    h5_3 = Dense(output_size, activation="relu", use_bias=True, kernel_initializer='he_normal', name='Dense_2')(h5_2)  # (layer_h15)

    outputs = Activation('softmax', name='Activation_1')(h5_3)
    base_model = keras.Model(inputs=inputs, outputs=outputs)

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # keras.Lambda(function, output_shape=None, mask=None, arguments=None)
    # 将任意表达式封装为 Layer 对象
    loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')([labels, outputs, input_length, label_length])
    ctc_model = keras.Model(inputs=[labels, inputs, input_length, label_length], outputs=loss_out)

    opt = keras.optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, decay=0.01, epsilon=10e-8)
    # ctc_model=multi_gpu_model(ctc_model,gpus=2)
    ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt, metrics=['accuracy'])

    return base_model, ctc_model

# # 交叉熵损失函数的CRNN
# def create_crnn_cross_model(input_size, output_size):
#     inputs = Input(name='the_inputs', shape=input_size)
#     # 1
#     h1_1 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_1-1')(inputs)
#     h1_2 = BatchNormalization(name='BatchNormal_1-1')(h1_1)
#     h1_3 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_1-2')(h1_2)
#     h1_4 = BatchNormalization(name='BatchNormal_1-2')(h1_3)
#     h1_5 = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", name='MaxPooling2D_1')(h1_4)
#     # 2
#     h2_1 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_2-1')(h1_5)
#     h2_2 = BatchNormalization(name='BatchNormal_2-1')(h2_1)
#     h2_3 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_2-2')(h2_2)
#     h2_4 = BatchNormalization(name='BatchNormal_2-2')(h2_3)
#     h2_5 = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", name='MaxPooling2D_2')(h2_4)
#     # # 3
#     # h3_1 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_3-1')(h2_5)
#     # h3_2 = BatchNormalization(name='BatchNormal_3-1')(h3_1)
#     # h3_3 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_3-2')(h3_2)
#     # h3_4 = BatchNormalization(name='BatchNormal_3-2')(h3_3)
#     # h3_5 = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", name='MaxPooling2D_3')(h3_4)
#     # # 4
#     # h4_1 = Conv2D(512, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_4-1')(h3_5)
#     # h4_2 = BatchNormalization(name='BatchNormal_4-1')(h4_1)
#     # h4_3 = Conv2D(512, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', name='Conv2D_4-2')(h4_2)
#     # h4_4 = BatchNormalization(name='BatchNormal_4-2')(h4_3)
#     # 由于声学模型网络结构原因（3个maxpooling层），我们的音频数据的每个维度需要能够被8整除。这里输入序列经过卷积网络后，长度缩短了8倍，因此我们训练实际输入的数据为wav_len//8。
#     # h5_1 = Reshape((-1, int(input_size[1]//8*512)), name='Reshape_1')(h4_4)

#     # lstm_1 = LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(h5_1)
#     # lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm_1)
#     h5_11 = Flatten()(h2_5)#(lstm_2)
#     h5_2 = Dense(512, activation='relu', use_bias=True, kernel_initializer='he_normal', name='Dense_1')(h5_11)
#     h5_3 = BatchNormalization(name='BatchNormal_5-1')(h5_2)
#     outputs = Dense(output_size, activation='softmax', name='the_labels')(h5_3)#(layer_h15)
#     base_model = keras.Model(inputs=inputs, outputs=outputs)

#     #opt = keras.optimizers.Adam(lr=0.1, beta_1=0.09, beta_2=0.999, decay=0.1, epsilon=10e-8)
#     opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

#     # ctc_model=multi_gpu_model(ctc_model, gpus=2)
#     base_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#     return base_model


# 交叉熵损失函数的CRNN
def create_crnn_cross_model(input_size, output_size):
    inputs = Input(name='the_inputs', shape=input_size)
    h1 = Conv2D(32, 3, activation='relu')(inputs)
    h1 = Conv2D(64, 3, activation='relu')(h1)
    block1_out = MaxPooling2D(3)(h1)

    h2 = Conv2D(64, 3, activation='relu', padding='same')(block1_out)
    h2 = Conv2D(64, 3, activation='relu', padding='same')(h2)
    block2_out = add([h2, block1_out])

    h3 = Conv2D(64, 3, activation='relu', padding='same')(block2_out)
    h3 = Conv2D(64, 3, activation='relu', padding='same')(h3)
    block3_out = add([h3, block2_out])

    h4 = Conv2D(64, 3, activation='relu')(block3_out)
    h4 = GlobalMaxPool2D()(h4)
    h4 = Dense(256, activation='relu')(h4)
    h4 = Dropout(0.5)(h4)
   
    outputs = Dense(output_size, activation='softmax', name='the_labels')(h4)#(layer_h15)
    base_model = keras.Model(inputs=inputs, outputs=outputs)

    #opt = keras.optimizers.Adam(lr=0.1, beta_1=0.09, beta_2=0.999, decay=0.1, epsilon=10e-8)
    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

    # ctc_model=multi_gpu_model(ctc_model, gpus=2)
    base_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return base_model



