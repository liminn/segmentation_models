import os
import cv2
import yaml
import math
import random
import numpy as np
from utils import utils
from keras.utils import Sequence

# set specific config file
cfg_path = "./configs/fpn.yaml"
with open(cfg_path) as fp:
    cfg = yaml.load(fp)

class TrainGen(Sequence):
    def __init__(self):
        filename = cfg["DATA"]["TRAIN_TXT_PATH"]
        with open(filename, 'r') as f:
            self.names = f.read().splitlines()
        np.random.shuffle(self.names)
        self.batch_size = cfg["TRAINNING"]["BATCH_SIZE"]
        self.img_rows = cfg["MODEL"]["INPUT_ROWS"]
        self.img_cols = cfg["MODEL"]["INPUT_COLS"]
        self.num_classes = cfg["MODEL"]["NUM_CLASSES"]
        self.img_path = cfg["DATA"]["IMAGE_PATH"]
        self.label_path = cfg["DATA"]["LABEL_PATH"]

    def __len__(self):
        return int(np.ceil(len(self.names) / float(self.batch_size)))

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_length = min(self.batch_size, (len(self.names) - i))
        batch_x = np.empty((batch_length, self.img_rows, self.img_cols, 3), dtype=np.float32)
        batch_y = np.empty((batch_length, self.img_rows, self.img_cols, self.num_classes), dtype=np.uint8)

        for i_batch in range(batch_length):
            # read image and mask(0~255)
            img_name = self.names[i]
            image_path = os.path.join(self.img_path, img_name)
            image = cv2.imread(image_path,1)
        
            img_name_prefix = img_name.split('.')[0]
            mask_name = img_name_prefix+".png"
            mask_path = os.path.join(self.label_path, mask_name)
            mask = cv2.imread(mask_path,0)

            # random crop
            patch_size = (self.img_rows,self.img_cols)
            image,mask = utils.random_patch(image,mask,patch_size)

            # random trimap: 0/255 -> 0/128/255
            trimap = utils.random_trimap(mask)
            
            # random horizontaly flip
            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                trimap = np.fliplr(trimap)

            # random rotation(90/180/270)
            if np.random.random_sample() > 0.5:
                # degree = random.randint(1,3)  # rotate 1/2/3 * 90 degrees
                degree = 1
                image = np.rot90(image, degree)
                trimap = np.rot90(trimap, degree)
            
            assert image.shape[:2] == trimap.shape[:2]

            # save the image-trimap patch
            pair_save_dir = cfg["CHECKPOINT"]["MODEL_DIR_BASE"]+'/'+cfg["CHECKPOINT"]["MODEL_DIR"]+'/'+cfg["CHECKPOINT"]["TRAIN_PAIR_DIR"]
            if not os.path.exists(pair_save_dir):
                os.makedirs(pair_save_dir)
            pair_image_name = img_name_prefix + '_image.png'
            pair_image_path = os.path.join(pair_save_dir,pair_image_name)
            cv2.imwrite(pair_image_path,image)
            pair_trimap_name = img_name_prefix + '_trimap.png'
            pair_trimap_path = os.path.join(pair_save_dir,pair_trimap_name)
            cv2.imwrite(pair_trimap_path,trimap)

            # normalize image
            image = image.astype(np.float32) / 255.0

            # trimap one-hot encoding
            trimap = utils.trimap_one_hot_encoding(trimap)
            
            batch_x[i_batch] = image
            batch_y[i_batch] = trimap
            

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)

class ValidGen(Sequence):
    def __init__(self):
        filename = cfg["DATA"]["VALID_TXT_PATH"]
        with open(filename, 'r') as f:
            self.names = f.read().splitlines()
        np.random.shuffle(self.names)
        #self.batch_size = cfg["TRAINNING"]["BATCH_SIZE"]
        self.batch_size = 1
        # self.img_rows = cfg["MODEL"]["INPUT_ROWS"]
        # self.img_cols = cfg["MODEL"]["INPUT_COLS"]
        self.num_classes = cfg["MODEL"]["NUM_CLASSES"]
        self.img_path = cfg["DATA"]["IMAGE_PATH"]
        self.label_path = cfg["DATA"]["LABEL_PATH"]

    def __len__(self):
        return int(np.ceil(len(self.names) / float(self.batch_size)))

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_length = min(self.batch_size, (len(self.names) - i))
        # batch_x = np.empty((batch_length, self.img_rows, self.img_cols, 3), dtype=np.float32)
        # batch_y = np.empty((batch_length, self.img_rows, self.img_cols, self.num_classes), dtype=np.uint8)

        for i_batch in range(batch_length):
            # read image and mask(0~255)
            img_name = self.names[i]
            image_path = os.path.join(self.img_path, img_name)
            image = cv2.imread(image_path,1)
            
            img_name_prefix = img_name.split('.')[0]
            mask_name = img_name_prefix+".png"
            mask_path = os.path.join(self.label_path, mask_name)
            mask = cv2.imread(mask_path,0)

            # # pad to fixed size
            # """
            #     待改：此处待去除
            #     因为：最好的valid方式一定是：batch为1，原尺寸输入，逐张进行valid
            #     需要：模型支持(None,None,3)的输入
            # """
            # patch_size = (self.img_rows,self.img_cols)
            # image,mask = pad_patch(image,mask,patch_size)

            # make sure h/w less than 1000
            rows, cols, _ = image.shape
            num_max = 512
            if(rows>num_max or cols>num_max):
                if rows>cols:
                    ratio = num_max/rows
                    rows_new = num_max
                    cols_new = int(ratio*cols)
                else:
                    ratio = num_max/cols
                    rows_new = int(ratio*rows)
                    cols_new = num_max
                image = cv2.resize(image, (cols_new, rows_new), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (cols_new,rows_new), interpolation=cv2.INTER_LINEAR)
                #print("change valid image shape:{}".format(image.shape))

            # make sure h/w be divisible by factor 32
            factor = 32
            rows, cols, _ = image.shape
            rows_new = rows//factor*factor
            cols_new = cols//factor*factor
            image = cv2.resize(image, (cols_new, rows_new), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (cols_new,rows_new), interpolation=cv2.INTER_LINEAR)
            #print("input valid image shape:{}".format(image.shape))

            # random trimap: 0/255 -> 0/128/255
            trimap = utils.random_trimap(mask)
            
            assert image.shape[:2] == trimap.shape[:2]
            
            # save the image-trimap patch
            pair_save_dir = cfg["CHECKPOINT"]["MODEL_DIR_BASE"]+'/'+cfg["CHECKPOINT"]["MODEL_DIR"]+'/'+cfg["CHECKPOINT"]["VALID_PAIR_DIR"]
            if not os.path.exists(pair_save_dir):
                os.makedirs(pair_save_dir)
            pair_image_name = img_name_prefix + '_image.png'
            pair_image_path = os.path.join(pair_save_dir,pair_image_name)
            cv2.imwrite(pair_image_path,image)
            pair_trimap_name = img_name_prefix + '_trimap.png'
            pair_trimap_path = os.path.join(pair_save_dir,pair_trimap_name)
            cv2.imwrite(pair_trimap_path,trimap)
            
            # normalize image
            image = image.astype(np.float32) / 255.0

            # trimap one-hot encoding
            trimap = utils.trimap_one_hot_encoding(trimap)
            
            batch_x = np.empty((batch_length, image.shape[0], image.shape[1], 3), dtype=np.float32)
            batch_y = np.empty((batch_length, image.shape[0], image.shape[1], self.num_classes), dtype=np.uint8)
            batch_x[i_batch] = image
            batch_y[i_batch] = trimap

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)

def train_gen():
    return TrainGen()

def valid_gen():
    return ValidGen()

if __name__ == '__main__':
    pass
