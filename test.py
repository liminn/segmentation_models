
import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from keras import backend as K

from utils import utils
from segmentation_models import FPN
# 待改
#from segmentation_models.pspnet_temp.pspnet_model import build_pspnet
from segmentation_models.pspnet_temp.pspnet_model_None import build_pspnet

if __name__ == '__main__':
    
    # set specific config file
    cfg_path = "./configs/fpn.yaml"
    with open(cfg_path) as fp:
        cfg = yaml.load(fp)

    # set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # set model
    if cfg["MODEL"]["MODEL_NAME"] == "PSPNet_temp":
        model = build_pspnet(cfg["MODEL"]["NUM_CLASSES"], resnet_layers=50, input_shape=(cfg["MODEL"]["INPUT_ROWS"],cfg["MODEL"]["INPUT_COLS"]))
        set_npy_weights(weights_path=cfg["MODEL"]["PT_PATH"], model = model)
    elif cfg["MODEL"]["MODEL_NAME"] == "PSPNet":
        model = PSPNet(backbone_name=cfg["MODEL"]["BACKBONE_NAME"],
                       input_shape=(cfg["MODEL"]["INPUT_ROWS"],cfg["MODEL"]["INPUT_COLS"], 3),
                       classes=cfg["MODEL"]["NUM_CLASSES"])
    elif cfg["MODEL"]["MODEL_NAME"] == "FPN":
        model = FPN(backbone_name=cfg["MODEL"]["BACKBONE_NAME"],
                       input_shape=(None,None,3),
                       classes=cfg["MODEL"]["NUM_CLASSES"])
    else:
        raise Exception("Error: do not support model:{}".format(cfg["MODEL"]["MODEL_NAME"]))

    # load model
    print(cfg)
    model.load_weights(cfg["TEST"]["CKPT_PATH"])

    # set test image txt
    filename = cfg["DATA"]["TEST_TXT_PATH"]
    with open(filename, 'r') as f:
        names = f.read().splitlines()

    # set test result path
    test_result_path = cfg["TEST"]["TEST_RESULT_DIR_BASE"] + '/' +cfg["CHECKPOINT"]["MODEL_DIR"]
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    for i in range(len(names)):
        name = names[i]
        image_path = os.path.join(cfg["DATA"]["IMAGE_PATH"], name)
        image = cv2.imread(image_path,1)
      
        # make sure h/w less than 1000
        rows, cols, _ = image.shape
        num_max = 1000
        if(rows>num_max or cols>num_max):
            if rows>cols:
                ratio = num_max/rows
                rows_new = 1000
                cols_new = int(ratio*cols)
            else:
                ratio = num_max/cols
                rows_new = int(ratio*rows)
                cols_new = 1000
            image = cv2.resize(image, (cols_new, rows_new), interpolation=cv2.INTER_LINEAR)
            #print("change valid image shape:{}".format(image.shape))

        # make sure h/w be divisible by factor 32
        factor = 32
        rows, cols, _ = image.shape
        rows_new = rows//factor*factor
        cols_new = cols//factor*factor
        image = cv2.resize(image, (cols_new, rows_new), interpolation=cv2.INTER_LINEAR)
        #print("input valid image shape:{}".format(image.shape))

        save_path = test_result_path+'/'+name.split('.')[0]+"_bgr.png"
        cv2.imwrite(save_path,image)
        
        x_test = np.empty((1, image.shape[0], image.shape[1], 3), dtype=np.float32)
        x_test[0, :, :, 0:3] = image/255.0
        
        out = model.predict(x_test)
        out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))
        # print(out.shape)
        out = np.argmax(out, axis=2)
        # print(0 in out)
        # print(1 in out)
        # print(2 in out)
        # print(out.shape)
        
        result_rgb = utils.colorful(out)
        save_path = test_result_path+'/'+name.split('.')[0]+"_result.png"
        cv2.imwrite(save_path,result_rgb)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        save_path = os.path.join(test_result_path,name.split('.')[0]+"_predict_compare.png")
        utils.vis_segmentation(image_rgb,result_rgb,save_path)
        print("generating: {}".format(save_path))

    K.clear_session()
    
