from __future__ import print_function
from math import ceil
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.optimizers import SGD
from keras.backend import tf as ktf
import keras.backend as K

import tensorflow as tf

import cv2
import numpy as np
import cv2 as cv

def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)

# class Interp(layers.Layer):

#     def __init__(self, new_size, **kwargs):
#         self.new_size = new_size
#         super(Interp, self).__init__(**kwargs)

#     def build(self, input_shape):
#         super(Interp, self).build(input_shape)

#     def call(self, inputs, **kwargs):
#         # new_height, new_width = self.new_size
#         #resized = ktf.image.resize_images(inputs, [new_height, new_width],
#         #                                   align_corners=True)
#         resized = tf.image.resize_bilinear(inputs, self.new_size)
#         #resized = ktf.image.resize_nearest_neighbor(inputs, self.new_size)
#         return resized

#     # def compute_output_shape(self, input_shape):
#     #     return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

#     def get_config(self):
#         config = super(Interp, self).get_config()
#         #config['new_size'] = self.new_size
#         config['new_size'] = (None,None)
#         return config

def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_reduce",
             "conv" + lvl + "_" + sub_lvl + "_1x1_reduce_bn",
             "conv" + lvl + "_" + sub_lvl + "_3x3",
             "conv" + lvl + "_" + sub_lvl + "_3x3_bn",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase_bn"]
    if modify_stride is False:
        prev = Conv2D(64 * level, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(64 * level, (1, 1), strides=(2, 2), name=names[0], use_bias=False)(prev)
    prev = BN(name=names[1])(prev)
    prev = Activation('relu')(prev)

    prev = ZeroPadding2D(padding=(pad, pad))(prev)
    prev = Conv2D(64 * level, (3, 3), strides=(1, 1), dilation_rate=pad, name=names[2], use_bias=False)(prev)
    prev = BN(name=names[3])(prev)
    prev = Activation('relu')(prev)

    prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[4], use_bias=False)(prev)
    prev = BN(name=names[5])(prev)
    return prev


def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_proj", "conv" + lvl + "_" + sub_lvl + "_1x1_proj_bn"]

    if modify_stride is False:
        prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(256 * level, (1, 1), strides=(2, 2), name=names[0], use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    return prev


def empty_branch(prev):
    return prev

# 对捷径(输入)进行下采样且升维后，然后再残差连接的模块
def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    prev_layer = Activation('relu')(prev_layer)
    # 卷积
    block_1 = residual_conv(prev_layer, level,
                            pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                            modify_stride=modify_stride)
    # 捷径
    block_2 = short_convolution_branch(prev_layer, level,
                                       lvl=lvl, sub_lvl=sub_lvl,
                                       modify_stride=modify_stride)
    added = Add()([block_1, block_2])
    return added

# 直接与捷径(输入)，进行残差连接的模块
def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
    prev_layer = Activation('relu')(prev_layer)
    # 卷积
    block_1 = residual_conv(prev_layer, level, pad=pad,
                            lvl=lvl, sub_lvl=sub_lvl)
    # 捷径
    block_2 = empty_branch(prev_layer)
    added = Add()([block_1, block_2])
    return added

# ResNet((473,473,3),50)
def ResNet(inp, layers):
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    # Short branch(only start of network)
    # ResNet起始的几层
    # 疑问：不应该只是一个7x7，s=2的卷积，接一个3x3，s=2的maxpooling么？
    # "conv1_1_3x3_s2"
    cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0], use_bias=False)(inp)   # 输出尺寸：237x237x64 
    bn1 = BN(name=names[1])(cnv1)         # "conv1_1_3x3_s2/bn"
    relu1 = Activation('relu')(bn1)       # "conv1_1_3x3_s2/relu"
    # "conv1_2_3x3"
    cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2], use_bias=False)(relu1) # 输出尺寸：237x237x64
    bn1 = BN(name=names[3])(cnv1)         # "conv1_2_3x3/bn"
    relu1 = Activation('relu')(bn1)       # "conv1_2_3x3/relu"
    # "conv1_3_3x3"
    cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4],use_bias=False)(relu1) # 输出尺寸：237x237x128
    bn1 = BN(name=names[5])(cnv1)         # "conv1_3_3x3/bn"
    relu1 = Activation('relu')(bn1)       # "conv1_3_3x3/relu"
    # "pool1_3x3_s2"
    res = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(2, 2))(relu1)                     # 输出尺寸：119x119x128 

    # ---Residual layers(body of network)
    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """
    # ResNet50和ResNet101的conv2_1-conv2_3的重复次数一致，都是3次
    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)                                           # 输出尺寸：119x119x256
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)                                   # 输出尺寸：119x119x256
    # ResNet50和ResNet101的conv3_1-conv3_3的重复次数一致，都是4次
    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)                       # 输出尺寸：60x60x512
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)                                   # 输出尺寸：60x60x512
    # ResNet50的conv4_1-conv4_3的重复次数为6次
    if layers is 50:
        # 4_1 - 4_6
        # 注意：此处步长原本为2，现在改为了1；此外，第二个3x3卷积施加了步长为2的空洞卷积                        # 原输出尺寸：30x30x1024
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)                                       # 现输出尺寸：60x60x1024            
        for i in range(5):
            # 注意：第二个3x3卷积全部变成了步长为2的空洞卷积 
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)                               # 输出尺寸： 60x60x1024       
    # ResNet101的conv4_1-conv4_3的重复次数为23次
    elif layers is 101:
        # 4_1 - 4_23
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)                                      
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")
    # ResNet50和ResNet101的conv5_1-conv5_3的重复次数一致，都是3次
    # 5_1 - 5_3
    # 注意：此处步长原本为2，现在改为了1；此外，第二个3x3卷积施加了步长为4的空洞卷积  
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)                                           # 输出尺寸：60x60x2048
    for i in range(2):
        # 注意：第二个3x3卷积全部变成了步长为4的空洞卷积 
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)                                   # 输出尺寸：60x60x2048

    res = Activation('relu')(res)
    return res

# prev_layer = res(res的维度(60,60,2048)),level = 1/2/3/6, feature_map_shape = (60,60), input_shape=(473,473)
def interp_block(prev_layer, level, feature_map_shape, input_shape):
    # if input_shape == (473, 473):
    #     kernel_strides_map = {1: 60, 2: 30, 3: 20, 6: 10}
    # elif input_shape == (320,320):                                  # 改变的地方！
    #     kernel_strides_map = {1: 40, 2: 20, 3: 10, 6: 5}            # 其实对应的结果应该为：1，2，4，8
    # elif input_shape == (713, 713):
    #     kernel_strides_map = {1: 90, 2: 45, 3: 30, 6: 15}
    # else:
    #     print("Pooling parameters for input shape ", input_shape, " are not defined.")
    #     exit(1)

    shape = K.int_shape(prev_layer)[1:3]

    # kernel_strides_map = {1: 88, 2: 44, 4: 22, 8: 11}
    kernel_strides_map = {1: 64, 2: 32, 4: 16, 8: 8}

    names = ["conv5_3_pool" + str(level) + "_conv", 
             "conv5_3_pool" + str(level) + "_conv_bn"]

    # 根据平均池化结果的空间维度，获取核的大小，步长的大小和核的大小相等
    # 如对于1x1的平均池化结果，核大小为(60,60)，步长为(60,60)
    # 如对于2x2的平均池化结果，核大小为(30,30)，步长为(30,30)
    # 如对于3x3的平均池化结果，核大小为(20,20)，步长为(20,20)
    # 如对于6x6的平均池化结果，核大小为(10,10)，步长为(10,10)
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    # 对于核大小为(60,60)，步长为(60,60)的平均池化，输出尺寸为1x1x2048
    # 对于核大小为(30,30)，步长为(30,30)的平均池化，输出尺寸为2x2x2048
    # 对于核大小为(20,20)，步长为(20,20)的平均池化，输出尺寸为3x3x2048
    # 对于核大小为(10,10)，步长为(10,10)的平均池化，输出尺寸为6x6x2048
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    # 降维，调整通道数为512
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    # prev_layer = Lambda(Interp, arguments={
    #                    'shape': feature_map_shape})(prev_layer)
    # 空间尺寸缩放回(60,60)
    # 则prev_layer的输出尺寸为：(60,60，512)
    #prev_layer = Interp(feature_map_shape)(prev_layer)
    
    def resize(x, size):
        #print("xxxxxxxx")
        #print(K.int_shape(prev_layer))
        #return tf.image.resize_bilinear(x, size)
        #return ktf.image.resize_images(x, size,align_corners=True)
        return tf.image.resize_bilinear(x, size, align_corners=True)

    prev_layer = Lambda(resize,output_shape=(shape[0],shape[1],512),arguments={'size':feature_map_shape})(prev_layer)
    # prev_layer = Interp(feature_map_shape)(prev_layer)
    return prev_layer

# res的维度(60,60,2048)，input_shape=(473,473)
def build_pyramid_pooling_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    # feature_map_size = (60,60)
    # 疑问：不就等于res的空间尺寸么，为啥还要再计算下
    # 答：也可以读res的空间尺寸，也可以通过input_shape计算，都行
    #feature_map_size = tuple(int(ceil(input_dim / 8.0)) for input_dim in input_shape)
    # feature_map_size = K.int_shape(res)[1:3]
    # print(feature_map_size)
    feature_map_size = K.shape(res)[1:3]
    #print("feature_map_size:{}".format(feature_map_size))

    #print(K.int_shape(feature_map_size))

    #print("PSP module will interpolate to a final feature map size of %s" % (feature_map_size, ))
    # 对res进行平均池化到1x1x2048，降维(1x1 conv,bn,relu)到1x1x512，缩放回60x60x512
    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)  # 输出尺寸为(60,60，512)
    # 对res进行平均池化到2x2x2048，降维(1x1 conv,bn,relu)到2x2x512，缩放回60x60x512
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)  # 输出尺寸为(60,60，512)
    # 对res进行平均池化到3x3x2048，降维(1x1 conv,bn,relu)到3x3x512，缩放回60x60x512
    interp_block3 = interp_block(res, 4, feature_map_size, input_shape)  # 输出尺寸为(60,60，512)
    # 对res进行平均池化到6x6x2048，降维(1x1 conv,bn,relu)到6x6x512，缩放回60x60x512
    interp_block6 = interp_block(res, 8, feature_map_size, input_shape)  # 输出尺寸为(60,60，512)

    # print(K.int_shape(interp_block1))
    # print(K.int_shape(interp_block2))
    # print(K.int_shape(interp_block3))
    # print(K.int_shape(interp_block6))

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    # 输入的res与四层池化结果相串联，输出尺寸为(60,60,(512x4+2048)) = (60,60,4096)
    res = Concatenate()([res,interp_block6,interp_block3,interp_block2,interp_block1]) # 输出尺寸为(60,60,4096)
    return res

# 例如: nb_classes=21,resnet_layers=50,input_shape=（473，473）
def build_pspnet(nb_classes, resnet_layers=50, input_shape=(473,473)):
    """Build PSPNet."""
    #print("Building a PSPNet based on ResNet %i expecting inputs of shape %s predicting %i classes" % (
    #    resnet_layers, input_shape, nb_classes))

    #inp = Input((None, None, 3))                       # 输入维度：(473,473,3)
    inp = Input((input_shape[0], input_shape[1], 3))  
    # print(input_shape[0])
    # print(input_shape[1])
    size = K.shape(inp)[1:3]
    #print(size)
    shape = K.int_shape(inp)

    # 即调用ResNet((473,473,3),layers=50)
    # 注意，施加膨胀卷积的ResNet50/101,即是：
    # 1.把conv4_x的第一个步长为2的1x1卷积置为1，其余所有的3x3卷积，都变为步长为2的膨胀卷积
    # 2.和conv5_x的第一个步长为2的1x1卷积置为1，其余所有的3x3卷积，都变为步长为4的膨胀卷积
    # 原Resnet,输入为(224,224,3)->输出为(7,7,2048)，              即进行了5次下采样，即输出为原图的1/32   output_dim = ceil(input_dim / 8.0) 此处的输出是指卷积层的输出，不算全连接层
    # 施加膨胀卷积后的ResNet,输入为(224,224,3)->输出为(28,28,2048)  即进行了3次下采样，即输出为原图的1/8    此处的输出是指卷积层的输出，不算全连接层
    # 原Resnet,输入为(473,473,3)->输出为(15,15,2048)   
    # 施加膨胀卷积后的ResNet,输入为(473,473,3)->输出为(60,60,2048)

    # 若输入变为input_shape=（320，320)，则输出为(40,40)
    res = ResNet(inp, layers=resnet_layers)                                   # 输出维度：(60,60,2048)
    psp = build_pyramid_pooling_module(res, input_shape)                      # 输出尺寸：(60,60,4096)

    # 1x1卷积，降维到512
    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",   # 输出尺寸：(60,60,512)
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # 1x1卷积，调整维度等于目标分类数num_class
    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)           # 输出尺寸：(60,60,num_class)
    # x = Lambda(Interp, arguments={'shape': (input_shape[0], input_shape[1])})(x)

    # 缩放空间尺寸到原始的输入空间尺寸
    # x = Interp([input_shape[0], input_shape[1]])(x)                           # 输出尺寸：(473,473,num_class)
    
    
    def resize(x, size):
        #return ktf.image.resize_images(x, size,align_corners=True)
        #return tf.image.resize_bilinear(x, tf.cast(size*2//2, dtype=tf.int32), align_corners=True)
        return tf.image.resize_bilinear(x, size, align_corners=True)
        #return tf.image.resize_bilinear(x, size)
    x = Lambda(resize,output_shape=(shape[1],shape[2],nb_classes),arguments={'size':size})(x)
    
    #size = K.shape(inp)[1:3]
    # prev_layer = Interp(feature_map_shape)(prev_layer)

    #x = Interp(size)(x)

    x = Activation('softmax')(x)

    model = Model(inputs=inp, outputs=x)

    return model

def colorful2(out):
    result_rgb = np.empty((out.shape[0], out.shape[1], 3), dtype=np.uint8)
    for row in range(out.shape[0]):
        for col in range(out.shape[1]):
            # 背景区域为第0类
            if(out[row,col]==0):
                result_rgb[row,col,0]=0
                result_rgb[row,col,1]=0
                result_rgb[row,col,2]=0
            # 前景区域为第1类
            if(out[row,col]==1):
                result_rgb[row,col,0]=255
                result_rgb[row,col,1]=255
                result_rgb[row,col,2]=255
            # 128区域为第2类
            if(out[row,col]==2): 
                result_rgb[row,col,0]=128
                result_rgb[row,col,1]=128
                result_rgb[row,col,2]=128
    return result_rgb

def colorful(out,shape_original):
    result_rgb = np.empty((img_rows, img_cols, 3), dtype=np.uint8)
    for row in range(img_rows):
        for col in range(img_cols):
            # 背景区域为第0类
            if(out[row,col]==0):
                result_rgb[row,col,0]=0
                result_rgb[row,col,1]=0
                result_rgb[row,col,2]=0
            # 前景区域为第1类
            if(out[row,col]==1):
                result_rgb[row,col,0]=255
                result_rgb[row,col,1]=255
                result_rgb[row,col,2]=255
            # 128区域为第2类
            if(out[row,col]==2): 
                result_rgb[row,col,0]=128
                result_rgb[row,col,1]=128
                result_rgb[row,col,2]=128
    return result_rgb

def get_image(name):
    image_path = os.path.join(rgb_image_path, name)
    # >0 Return a 3-channel color image.
    # =0 Return a grayscale image.
    # <0 Return the loaded image as is (with alpha channel).
    image = cv.imread(image_path,1)
    if image.shape != (img_rows, img_cols): # ?
        # dsize即指的是Size(width，height)
        image = cv.resize(image, dsize=(img_rows, img_cols), interpolation=cv.INTER_NEAREST)    
    return image

if __name__ == '__main__':
    with tf.device("/cpu:0"):
        t_net_weights_path = 'models/pspnet50_model.32-0.1993.hdf5'
        #pspnet_model = build_pspnet(3, resnet_layers=50, input_shape=(None,None))
        pspnet_model = build_pspnet(3, resnet_layers=50, input_shape=(None,None))
        pspnet_model.summary()
        #pspnet_model.load_weights(t_net_weights_path)
        print(1)
        # image_path = "./1_supervisely1826.jpg.jpg"
        # image_original = cv2.imread(image_path,1)
        # image_original = cv2.resize(image_original, dsize=(473, 473), interpolation=cv2.INTER_NEAREST) 
        # image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
        # shape = image_original.shape
        # print("inpout shape:{}".format(shape))
        # #cv2.imshow("image",image_original)
        # #cv2.waitKey(0)
        # x_test = np.empty((1, shape[0], shape[1], 3), dtype=np.float32)
        # x_test[0, :, :, 0:3] = image_original/255.0
        # out = pspnet_model.predict(x_test)
        # print("output shape:{}".format(out.shape))
        # out = np.reshape(out, (shape[0], shape[1], 3))
        # print("resized output shape:{}".format(out.shape))
        # out = np.argmax(out, axis=2)
  
        # result_rgb = colorful(out)
        # cv2.imwrite("./output.png",result_rgb)

        # img_rows = 473
        # img_cols = 473
        # num_classes = 3

        # image_original = cv.imread(image_path,1)
        # image_original_rgb = cv.cvtColor(image_original, cv.COLOR_BGR2RGB)
        # shape_original = image_original.shape

        # image_bgr = None
        # if image_original_rgb.shape != (img_rows, img_cols): # ?
        #     # dsize即指的是Size(width，height)
        #     image_bgr = cv.resize(image_original_rgb, dsize=(img_rows, img_cols), interpolation=cv.INTER_NEAREST)    

        # image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        # x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
        # x_test[0, :, :, 0:3] = image_rgb/255.0

        # out = pspnet_model.predict(x_test)
        # out = np.reshape(out, (img_rows, img_cols, num_classes))
        # out = np.argmax(out, axis=2)

        # result_rgb = colorful(out,shape_original)
        # # img.shape->(rows,cols,channels) 
        # # dsize=(x/cols,y/rows)
        # result_rgb_shape_original = cv.resize(result_rgb, dsize=(shape_original[1], shape_original[0]), interpolation=cv.INTER_NEAREST)    

        # cv2.imwrite("./output2.png",result_rgb_shape_original)


    K.clear_session()
