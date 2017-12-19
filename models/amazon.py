# -*- coding: utf-8 -*-
import os
import cv2
import string
import numpy as np

from keras import backend as K
from keras.layers import Input
from keras.layers import Reshape, Dense, Dropout, Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GRU, LSTM
from keras.layers import Add, Concatenate

from keras.models import Model, load_model


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def create_model(width, height, n_len, n_class, rnn_size=128, dropout_rate=0.25):
    input_tensor = Input((width, height, 3))
    x = input_tensor
    for i in range(3):
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

    x = Dense(units=32, activation='relu')(x)

    # 以下可以用 LSTM 替换。
    gru_1 = GRU(units=rnn_size, return_sequences=True, kernel_initializer='he_normal',
                recurrent_initializer='he_normal',
                name='gru1')(x)
    gru_1b = GRU(units=rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                 recurrent_initializer='he_normal', name='gru1_b')(x)
    gru1_merged = Add()([gru_1, gru_1b])

    gru_2 = GRU(units=rnn_size, return_sequences=True, kernel_initializer='he_normal',
                recurrent_initializer='he_normal',
                name='gru2')(gru1_merged)
    gru_2b = GRU(units=rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                 recurrent_initializer='he_normal', name='gru2_b')(gru1_merged)
    x = Concatenate()([gru_2, gru_2b])

    x = Dropout(dropout_rate)(x)
    x = Dense(units=n_class, kernel_initializer='he_normal', activation='softmax')(x)

    base_model = Model(inputs=input_tensor, outputs=x)

    labels = Input(name='the_labels', shape=[n_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    return base_model, model


def preprocess_image(width, height, captcha_path):
    X = np.zeros((1, width, height, 3), dtype=np.uint8)

    X[0] = cv2.imread(captcha_path).transpose(1, 0, 2)

    # TODO 要对输入的图片进行 Resize
    # 按道理从亚马逊验证码页面下载回来的图片，都是 200x70 规格的，只是为了做防御式编程考虑的。

    return X


def inference(model, characters, n_len, X_input):
    y_pred = model.predict(X_input)
    y_pred = y_pred[:, 2:, :]

    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :n_len]
    out = ''.join([characters[x] for x in out[0]])

    # TODO 有可能返回不规整数据
    return out


def main():
    characters = string.ascii_uppercase
    width, height, n_len, n_class = 200, 70, 6, len(characters) + 1

    # base_model, model = create_model(width, height, n_len, n_class)

    pretrained_model_path = os.path.join(os.path.dirname(__file__), '..', 'pretrained', 'amazon_20171106.h5')
    base_model = load_model(pretrained_model_path)
    # model.load_weights(pretrained_model_path)
    # base_model.load_weights(pretrained_model_path)

    captcha_test_dir = os.path.join(os.path.dirname(__file__), '..', 'captcha_images', 'amazon')
    captcha_path = os.path.join(captcha_test_dir, 'Captcha_oyavokqczd.jpg')
    # captcha_path = os.path.join(captcha_test_dir, 'success_HPNBKM.jpg')
    # captcha_path = os.path.join(captcha_test_dir, 'error_MRHGCN.jpg')

    X_test = preprocess_image(width, height, captcha_path)
    out = inference(base_model, characters, n_len, X_test)
    print(out)


if __name__ == '__main__':
    main()
