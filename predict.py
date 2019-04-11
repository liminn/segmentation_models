# import the necessary packages
import os
import random

import cv2
import keras.backend as K
import numpy as np

from config import img_rows,img_cols,num_classes,rgb_image_path,inference_output_path_base
#from pspnet_model import build_pspnet
#from pspnet_model_None import build_pspnet
from utils import colorful,vis_segmentation

import matplotlib.pyplot as plt
from matplotlib import gridspec

from segmentation_models import FPN

def pad_and_resize_to_target_size(image, target_rows, target_cols):
    rows,cols = image.shape[0],image.shape[1]
    if rows>cols:
        mat = np.zeros((rows, rows, 3), np.float32)
    else:
        mat = np.zeros((cols, cols, 3), np.float32)
    mat[0:rows, 0:cols,:] = image
    mat = cv2.resize(mat,(target_cols,target_rows),interpolation=cv2.INTER_LINEAR)
    return mat.astype(np.uint8)

if __name__ == '__main__':
    
    # 指定GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 加载模型
    # for id photo
    # model_weights_path = 'models/pspnet50_512x512_20190225_shm_id_data/pspnet50.21-0.0607.hdf5'
    # for person
    model_weights_path = 'models/fpn50_800x800_supervisely_20190329/model-68-0.1720.hdf5'
    # #model_weights_path = 'models/pspnet50_512x512_20190220_supervisely/pspnet50.29-0.2854.hdf5'
    # #pspnet50_model = build_pspnet(num_classes, resnet_layers=50, input_shape=(img_rows,img_cols))
    # pspnet_model = build_pspnet(3, resnet_layers=101, input_shape=(None,None))
    # pspnet_model.load_weights(model_weights_path)
    # pspnet_model.summary()
    BACKBONE = "resnet50"
    ACTIVATION='sigmoid'
    model = FPN(backbone_name=BACKBONE,
                input_shape=(img_rows, img_cols, 3),
                classes=num_classes,
                activation=ACTIVATION,
                encoder_weights='imagenet',
                pyramid_dropout=0.5)
    model.load_weights(model_weights_path)
    
    # 测试图片
    # for id photo
    #img_path_test = '/home/datalab/ex_disk1/bulang/SemanticHumanMatting/test_id_photo_20190130_2'
    # for person
    #img_path_test = "./data_total_1/image"
    img_path_test = "/home/datalab/ex_disk1/bulang/segmentation_data_repository/原始数据/supervisely/src_rename"
    # img_path_test = '/home/datalab/ex_disk1/bulang/SemanticHumanMatting/data_shm_person/Training_set/composite'
    # names = [f for f in os.listdir(img_path_test) if
    #                os.path.isfile(os.path.join(img_path_test, f)) and f.endswith('.png')]
    #print(names)
    filename = 'test_names.txt'
    with open(filename, 'r') as f:
        names = f.read().splitlines()

    # 定义推理输出路径
    # for id photo
    #output_dir = "test_output_20190226__shm_id_data_2"
    # for person
    output_dir = "test_supervisely_20190331"
    inference_out_path = os.path.join(inference_output_path_base,output_dir)
    if not os.path.exists(inference_out_path):
        os.makedirs(inference_out_path)

    for i in range(len(names)):
        name = names[i]

        image_path = os.path.join(img_path_test, name)
        image_bgr = cv2.imread(image_path,1)

        #image_bgr = pad_and_resize_to_target_size(image_bgr,img_cols,img_rows)
        image_bgr = cv2.resize(image_bgr,(img_cols,img_rows))
        cv2.imwrite(inference_out_path+'/'+str(i)+"_image_bgr.png",image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
        x_test[0, :, :, 0:3] = image_bgr/255.0

        out = model.predict(x_test)
        print(out.shape)
        out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))
        out = np.argmax(out, axis=2)

        result_rgb = colorful(out)

        # 同时展示原图/预测结果/预测结果叠加原图，并保存
        vis_seg_save_path = os.path.join(inference_out_path,name.split('.')[0]+"_predict_compare.png")
        vis_segmentation(image_rgb,result_rgb,vis_seg_save_path)
        print("generating: {}".format(vis_seg_save_path))

    K.clear_session()
    
