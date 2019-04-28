
import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from utils import utils
from segmentation_models import FPN
# 待改
#from segmentation_models.pspnet_temp.pspnet_model import build_pspnet
from segmentation_models.pspnet_temp.pspnet_model_None import build_pspnet

if __name__ == '__main__':
    
    # set specific config file
    cfg_path = "./configs/pspnet_temp.yaml"
    with open(cfg_path) as fp:
        cfg = yaml.load(fp)

    # set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # set model
    model = build_pspnet(cfg["MODEL"]["NUM_CLASSES"], cfg["MODEL"]["RESNET_LAYERS"], input_shape=(None,None))

    # load model
    model_weights_path = './checkpoint/PSPNet_temp_20190415/model-35-0.0704.hdf5'
    model.load_weights(model_weights_path)

    # set test image txt
    filename = cfg["DATA"]["TEST_TXT_PATH"]
    with open(filename, 'r') as f:
        names = f.read().splitlines()

    # set test result path
    dir_name = model_weights_path.split('/')[2]
    test_result_path = os.path.join(cfg["TEST"]["TEST_RESULT_PATH"],dir_name)
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    for i in range(len(names)):
        name = names[i]
        image_path = os.path.join(cfg["DATA"]["IMAGE_PATH"], name)
        image_bgr = cv2.imread(image_path,1)
        img_rows = cfg["MODEL"]["INPUT_ROWS"]
        img_cols = cfg["MODEL"]["INPUT_COLS"]
        
        #image_bgr = utils.pad_and_resize_to_target_size(image_bgr,img_rows,img_cols)
        #不支持
        image_bgr = utils.resize_to_target_size(image_bgr,img_rows,img_cols)
        save_path = test_result_path+'/'+name.split('.')[0]+"_bgr.png"
        cv2.imwrite(save_path,image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        x_test = np.empty((1, image_bgr.shape[0], image_bgr.shape[1], 3), dtype=np.float32)
        x_test[0, :, :, 0:3] = image_bgr/255.0
        
        out = model.predict(x_test)
        out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))
        out = np.argmax(out, axis=2)
        
        result_rgb = utils.colorful(out)
        save_path = os.path.join(test_result_path,name.split('.')[0]+"_predict_compare.png")
        utils.vis_segmentation(image_rgb,result_rgb,save_path)
        print("generating: {}".format(save_path))

    K.clear_session()
    
