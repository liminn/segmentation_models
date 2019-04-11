# import the necessary packages
import os
import random

import cv2
import keras.backend as K
import numpy as np

from model_None import build_pspnet

if __name__ == '__main__':
    
    ###
    # PSPNet支持任意尺度的输入，并给出任意尺度的输出
    # 因为，PSPNet中的上采样方式，均为resize回输入尺寸，故不存在像素数目丢失问题
    ###

    # model
    model_weights_path = 'models/pspnet50_model.32-0.1993.hdf5'
    #pspnet50_model = build_pspnet(num_classes, resnet_layers=50, input_shape=(img_rows,img_cols))
    pspnet50_model = build_pspnet(3, resnet_layers=50, input_shape=(None,None))
    #pspnet50_model = build_pspnet(num_classes, resnet_layers=50, input_shape=(473,473))
    pspnet50_model.load_weights(model_weights_path)
    #pspnet50_model.summary()

    # image
    image_path = "test_results_None/1.jpg"
    image_bgr = cv2.imread(image_path,1)

    # x_test
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_rows, img_cols = 320,320
    x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
    image_rgb_resize = cv2.resize(image_rgb,(img_cols,img_rows),cv2.INTER_LINEAR)
    x_test[0, :, :, 0:3] = image_rgb_resize/255.0

    # predict
    # out:(1,rows,cols,3),0/1/2
    out = pspnet50_model.predict(x_test)   
    out = np.reshape(out, (out.shape[1],out.shape[2],out.shape[3]))
    out = np.argmax(out, axis=2)

    # colorful
    result_rgb = np.zeros(out.shape, dtype=np.uint8)
    result_rgb = (out==0) * 0 + (out==1) * 255 + (out==2)*128
    
    # save
    save_path = "test_results_None/1_"+str(img_rows)+"x"+str(img_cols)+".png"
    cv2.imwrite(save_path, result_rgb)
    print("generating: {}".format(save_path))
    
    K.clear_session()

