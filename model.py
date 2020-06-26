# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:55:53 2020

@author: klal1
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D,BatchNormalization, Dropout
from keras.layers import Conv2DTranspose, Concatenate, LeakyReLU

def generator():
    inp_img = Input((1152, 768, 1))
    
    # DownSampling
    lay_1 = Conv2D(64, 5, padding='same', name='block1_conv')(inp_img)
    lay_1 = LeakyReLU(alpha=0.2)(lay_1)
    max_1 = MaxPool2D(name='block1_pool')(lay_1)
    
    lay_2 = Conv2D(128, 3, padding='same', name='block2_conv')(max_1)
    lay_2 = LeakyReLU(alpha=0.2)(lay_2)
    dout_2 = Dropout(0.1, name='block2_dropout')(lay_2)
    max_2 = MaxPool2D(name='block2_pool')(dout_2)
    bat_2 = BatchNormalization(name='block2_batchNorm')(max_2)
    
    lay_3 = Conv2D(128, 3, padding='same', name='block3_conv')(bat_2)
    lay_3 = LeakyReLU(alpha=0.2)(lay_3)
    dout_3 = Dropout(0.1, name='block3_dropout')(lay_3)
    max_3 = MaxPool2D(name='block3_pool')(dout_3)
    bat_3 = BatchNormalization(name='block3_batchNorm')(max_3)
    
    lay_4 = Conv2D(256, 3, padding='same', name='block4_conv')(bat_3)
    lay_4 = LeakyReLU(alpha=0.2)(lay_4)    
    dout_4 = Dropout(0.1, name='block4_dropout')(lay_4)
    max_4 = MaxPool2D(name='block4_pool')(dout_4)
    bat_4 = BatchNormalization(name='block4_batchNorm')(max_4)
    
    lay_5 = Conv2D(256, 3, padding='same', name='block5_conv')(bat_4)
    lay_5 = LeakyReLU(alpha=0.2)(lay_5)
    dout_5 = Dropout(0.1, name='block5_dropout')(lay_5)
    max_5 = MaxPool2D(name='block5_pool')(dout_5)
    bat_5 = BatchNormalization(name='block5_batchNorm')(max_5)
    
    lay_6 = Conv2D(512, 3, padding='same', name='block6_conv')(bat_5)
    lay_6 = LeakyReLU(alpha=0.2)(lay_6)    
    dout_6 = Dropout(0.1, name='block6_dropout')(lay_6)
    max_6 = MaxPool2D(name='block6_pool')(dout_6)
    bat_6 = BatchNormalization(name='block6_batchNorm')(max_6)
    
    lay_7 = Conv2D(1024, 3, padding='same', name='block7_conv')(bat_6)
    lay_7 = LeakyReLU(alpha=0.2)(lay_7)
    
    #UpSampling
    upS_8 = Conv2DTranspose(256, 3, strides=2, padding='same', name='block8_convTranspose')(lay_7)
    upS_8 = LeakyReLU(alpha=0.2)(upS_8)
    bat_8 = BatchNormalization(name='block8_batchNorm')(upS_8)
    con_8 = Concatenate(name='block8_concat')([bat_8, max_5])
    
    upS_9 = Conv2DTranspose(256, 3, strides=2, padding='same', name='block9_convTranspose')(con_8)
    upS_9 = LeakyReLU(alpha=0.2)(upS_9)
    bat_9 = BatchNormalization(name='block9_batchNorm')(upS_9)
    con_9 = Concatenate(name='block9_concat')([bat_9, max_4])
    
    upS_10 = Conv2DTranspose(128, 3, strides=2, padding='same', name='block10_convTranspose')(con_9)
    upS_10 = LeakyReLU(alpha=0.2)(upS_10)
    bat_10 = BatchNormalization(name='block10_batchNorm')(upS_10)
    con_10 = Concatenate(name='block10_concat')([bat_10, max_3])
    
    upS_11 = Conv2DTranspose(128, 3, strides=2, padding='same', name='block11_convTranspose')(con_10)
    upS_11 = LeakyReLU(alpha=0.2)(upS_11)
    bat_11 = BatchNormalization(name='block11_batchNorm')(upS_11)
    con_11 = Concatenate(axis=-1, name='block11_concat')([bat_11, max_2])
    
    upS_12 = Conv2DTranspose(64, 3, strides=2, padding='same', name='block12_convTranspose')(con_11)
    upS_12 = LeakyReLU(alpha=0.2)(upS_12)
    bat_12 = BatchNormalization(name='block12_batchNorm')(upS_12)
    con_12 = Concatenate(axis=-1, name='block12_concat')([bat_12, max_1])
    
    out = Conv2DTranspose(3, 3, strides=2, activation='tanh', padding='same',  name='block13_convTranspose')(con_12)
    return Model(inp_img, out)

def discriminator():
    img_A = Input((1152, 768, 1))
    img_B = Input((1152, 768, 3))
    
    lay_1_A = Conv2D(3, 3, padding='same', name='block1_conv_A')(img_A)
    lay_1_A = LeakyReLU(alpha=0.2)(lay_1_A)    
    lay_1_B = Conv2D(3, 3, padding='same', name='block1_conv_B')(img_B)
    lay_1_B = LeakyReLU(alpha=0.2)(lay_1_B)
    con_2 = Concatenate(axis=-1, name='block1_concat')([lay_1_A, lay_1_B])
    
    lay_2 = Conv2D(64, 3, padding='same', name='block2_conv')(con_2)
    lay_2 = LeakyReLU(alpha=0.2)(lay_2)    
    max_2 = MaxPool2D(name='block2_pool')(lay_2)
    bat_2 = BatchNormalization(name='block2_batchNorm')(max_2)
    
    lay_3 = Conv2D(128, 3, padding='same', name='block3_conv')(bat_2)
    lay_3 = LeakyReLU(alpha=0.2)(lay_3)
    max_3 = MaxPool2D(name='block3_pool')(lay_3)
    bat_3 = BatchNormalization(name='block3_batchNorm')(max_3)
    
    lay_4 = Conv2D(128, 3, padding='same', name='block4_conv')(bat_3)
    lay_4 = LeakyReLU(alpha=0.2)(lay_4)    
    max_4 = MaxPool2D(name='block4_pool')(lay_4)
    bat_4 = BatchNormalization(name='block4_batchNorm')(max_4)
    
    lay_5 = Conv2D(256, 3, padding='same', name='block5_conv')(bat_4)
    lay_5 = LeakyReLU(alpha=0.2)(lay_5)
    max_5 = MaxPool2D(name='block5_pool')(lay_5)
    bat_5 = BatchNormalization(name='block5_batchNorm')(max_5)
    
    lay_6 = Conv2D(256, 3, padding='same', name='block6_conv')(bat_5)
    lay_6 = LeakyReLU(alpha=0.2)(lay_6)
    max_6 = MaxPool2D(name='block6_pool')(lay_6)
    bat_6 = BatchNormalization(name='block6_batchNorm')(max_6)
    
    lay_7 = Conv2D(256, 3, padding='same', name='block7_conv')(bat_6)
    lay_7 = LeakyReLU(alpha=0.2)(lay_7)
    max_7 = MaxPool2D(name='block7_pool')(lay_7)
    bat_7 = BatchNormalization(name='block7_batchNorm')(max_7)
    validity = Conv2D(1, 1, activation='sigmoid', padding='same', name='block8_conv')(bat_7)
    return Model([img_A, img_B], validity)


def cGAN(gen, dis):    
    grey_img = Input((1152, 768,1))
    color_img = Input((1152, 768,3))

    generated_img = gen(grey_img)

    dis.trainable = False

    validity = dis([grey_img, generated_img])
    
    cgan = Model(inputs=[grey_img], outputs=[validity, generated_img])
    return cgan