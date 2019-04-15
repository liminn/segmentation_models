import os
import cv2
import yaml
import math
import random
import numpy as np
from utils import utils
from keras.utils import Sequence

# Set specific config file
cfg_path = "./configs/pspnet_temp.yaml"
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
        # 每一个epoch，从前往后读取self.ids，依据id，读取self.names
        # idx应为第几个batch，i为该次batch的起始点
        i = idx * self.batch_size
        # length为当前batch的大小
        length = min(self.batch_size, (len(self.names) - i))
        batch_x = np.empty((length, self.img_rows, self.img_cols, 3), dtype=np.float32)
        batch_y = np.empty((length, self.img_rows, self.img_cols, self.num_classes), dtype=np.uint8)

        for i_batch in range(length):
            img_name = self.names[i]
            image_path = os.path.join(self.img_path, img_name)
            image = cv2.imread(image_path,1)
        
            img_name_prefix = img_name.split('.')[0]
            mask_name = img_name_prefix+".png"
            mask_path = os.path.join(self.label_path, mask_name)
            mask = cv2.imread(mask_path,0)

            # 随机缩放
            image, mask = utils.random_rescale_image_and_mask(image,mask)

            # 随机旋转并剪裁
            if np.random.random_sample() > 0.5:
                if np.random.random_sample() > 0.5:
                    image_height, image_width = image.shape[0:2]
                    angle = random.randint(0,10)
                    image_rotated = utils.rotate_image(image, angle)
                    image_rotated_cropped = utils.crop_around_center(
                        image_rotated,
                        *utils.largest_rotated_rect(
                            image_width,
                            image_height,
                            math.radians(angle)
                        )
                    )
                    image = image_rotated_cropped
                    alpha_rotated = utils.rotate_image(mask, angle)
                    alpha_rotated_cropped = utils.crop_around_center(
                        alpha_rotated,
                        *utils.largest_rotated_rect(
                            image_width,
                            image_height,
                            math.radians(angle)
                        )
                    )
                    mask = alpha_rotated_cropped
                else:
                    image_height, image_width = image.shape[0:2]
                    angle = random.randint(350,360)
                    image_rotated = utils.rotate_image(image, angle)
                    image_rotated_cropped = utils.crop_around_center(
                        image_rotated,
                        *utils.largest_rotated_rect(
                            image_width,
                            image_height,
                            math.radians(angle)
                        )
                    )
                    image = image_rotated_cropped
                    alpha_rotated = utils.rotate_image(mask, angle)
                    alpha_rotated_cropped = utils.crop_around_center(
                        alpha_rotated,
                        *utils.largest_rotated_rect(
                            image_width,
                            image_height,
                            math.radians(angle)
                        )
                    )
                    mask = alpha_rotated_cropped
            
            # 实时处理alpha，得到trimap:128/0/255
            trimap = utils.generate_random_trimap(mask)
           
            # 定义随机剪裁尺寸
            crop_size = (self.img_rows,self.img_cols)
            # # 获得剪裁的起始点，其目的是为了保证剪裁的图像中包含未知像素
            x, y = utils.random_choice(trimap, crop_size)      
            # # x = random.randint(0,image.shape[1])
            # # y = random.randint(0,image.shape[0])

            # 剪裁image，到指定剪裁尺寸crop_size
            image = utils.safe_crop(image, x, y, crop_size)
            # 剪裁trimap，到指定剪裁尺寸crop_size
            trimap = utils.safe_crop(trimap, x, y, crop_size)

            # 随机翻转
            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                trimap = np.fliplr(trimap)
            
            #save the image/trimap crop patch
            # patch_save_dir = "show_data_loader_20190415"
            # if not os.path.exists(patch_save_dir):
            #     os.makedirs(patch_save_dir)
            # image_patch_path = patch_save_dir + '/' + img_name_prefix + '_image_' + str(i_batch) + '.png'
            # trimap_patch_path = patch_save_dir + '/' + img_name_prefix + '_trimap_' + str(i_batch) + '.png'
            # cv2.imwrite(image_patch_path,image)
            # cv2.imwrite(trimap_patch_path,trimap)

            batch_x[i_batch] = image/255.0
            batch_y[i_batch] = utils.make_trimap_for_batch_y(trimap) 

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)

class ValDataGen(Sequence):
    def __init__(self):
        filename = cfg["DATA"]["VALID_TXT_PATH"]
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
        # 每一个epoch，从前往后读取self.ids，依据id，读取self.names
        # idx应为第几个batch，i为该次batch的起始点
        i = idx * self.batch_size
        # length为当前batch的大小
        length = min(self.batch_size, (len(self.names) - i))
        batch_x = np.empty((length, self.img_rows, self.img_cols, 3), dtype=np.float32)
        batch_y = np.empty((length, self.img_rows, self.img_cols, self.num_classes), dtype=np.uint8)

        for i_batch in range(length):
            ###normal
            img_name = self.names[i] 
            image_path = os.path.join(self.img_path, img_name)
            image = cv2.imread(image_path,1)

            img_name_prefix = img_name.split('.')[0]
            mask_name = img_name_prefix+".png"
            mask_path = os.path.join(self.label_path, mask_name)
            mask = cv2.imread(mask_path,0)

            # 实时处理alpha，得到trimap:128/0/255
            trimap = utils.generate_random_trimap(mask)

            image = utils.pad_and_resize_to_target_size(image,self.img_rows, self.img_cols)
            trimap = utils.pad_and_resize_mask_to_target_size(trimap,self.img_rows, self.img_cols)

            # patch_save_dir = "show_data_loader_valid_20190415"
            # if not os.path.exists(patch_save_dir):
            #     os.makedirs(patch_save_dir)
            # image_patch_path = patch_save_dir + '/' + img_name_prefix + '_image_' + str(i_batch) + '.png'
            # trimap_patch_path = patch_save_dir + '/' + img_name_prefix + '_trimap_' + str(i_batch) + '.png'
            # cv2.imwrite(image_patch_path,image)
            # cv2.imwrite(trimap_patch_path,trimap)

            batch_x[i_batch] = image/255.0
            batch_y[i_batch] = utils.make_trimap_for_batch_y(trimap) 
            
            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)

def train_gen():
    return TrainGen()

def valid_gen():
    return ValDataGen()

if __name__ == '__main__':
    pass
