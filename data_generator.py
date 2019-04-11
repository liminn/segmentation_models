import os
import random
from random import shuffle
import math
import cv2
import numpy as np

from keras.utils import Sequence

from config import img_rows, img_cols, batch_size, num_classes,rgb_image_path, mask_img_path,unknown_code
from utils import generate_random_trimap, random_choice, safe_crop, make_trimap_for_batch_y,random_rescale_image_and_mask,make_mask_for_batch_y
from utils import rotate_image, largest_rotated_rect, crop_around_center
from utils import pad_and_resize_to_target_size,pad_and_resize_mask_to_target_size

class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage
        # filename会是"train_names.txt"或"valid_names.txt"
        # "train_names.txt"、"valid_names.txt"中图片名，例如"0-100.png"
        filename = '{}_names.txt'.format(usage)
        with open(filename, 'r') as f:
            self.names = f.read().splitlines()
        np.random.shuffle(self.names)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        # 每一个epoch，从前往后读取self.ids，依据id，读取self.names
        # idx应为第几个batch，i为该次batch的起始点
        i = idx * batch_size
        # length为当前batch的大小
        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 3), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols, num_classes), dtype=np.uint8)

        for i_batch in range(length):
            ###normal
            img_name = self.names[i] # xx.jpg
            image_path = os.path.join(rgb_image_path, img_name)
            image = cv2.imread(image_path,1)
            # img_name_prefix,useless = os.path.splitext(img_name)
            #mask_name = img_name_prefix+'.png'
            # base_name = os.path.basename(name)
            #img_name_prefix = img_name.split('_split_')[0]
            img_name_prefix = img_name.split('.')[0]
            mask_name = img_name_prefix+".png"
            mask_path = os.path.join(mask_img_path, mask_name)
            mask = cv2.imread(mask_path,0)

            image = cv2.resize(image, dsize=(img_rows, img_cols), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=(img_rows, img_cols), interpolation=cv2.INTER_NEAREST)

            """
            # 随机缩放
            image, mask = random_rescale_image_and_mask(image,mask)

            # 随机旋转并剪裁
            if np.random.random_sample() > 0.5:
                if np.random.random_sample() > 0.5:
                    image_height, image_width = image.shape[0:2]
                    angle = random.randint(0,10)
                    image_rotated = rotate_image(image, angle)
                    image_rotated_cropped = crop_around_center(
                        image_rotated,
                        *largest_rotated_rect(
                            image_width,
                            image_height,
                            math.radians(angle)
                        )
                    )
                    image = image_rotated_cropped
                    alpha_rotated = rotate_image(mask, angle)
                    alpha_rotated_cropped = crop_around_center(
                        alpha_rotated,
                        *largest_rotated_rect(
                            image_width,
                            image_height,
                            math.radians(angle)
                        )
                    )
                    mask = alpha_rotated_cropped
                else:
                    image_height, image_width = image.shape[0:2]
                    angle = random.randint(350,360)
                    image_rotated = rotate_image(image, angle)
                    image_rotated_cropped = crop_around_center(
                        image_rotated,
                        *largest_rotated_rect(
                            image_width,
                            image_height,
                            math.radians(angle)
                        )
                    )
                    image = image_rotated_cropped
                    alpha_rotated = rotate_image(mask, angle)
                    alpha_rotated_cropped = crop_around_center(
                        alpha_rotated,
                        *largest_rotated_rect(
                            image_width,
                            image_height,
                            math.radians(angle)
                        )
                    )
                    mask = alpha_rotated_cropped
            
            # 实时处理alpha，得到trimap:128/0/255
            trimap = generate_random_trimap(mask)
           
            # 定义随机剪裁尺寸
            crop_size = (img_rows,img_cols)
            # # 获得剪裁的起始点，其目的是为了保证剪裁的图像中包含未知像素
            x, y = random_choice(trimap, crop_size)      
            # # x = random.randint(0,image.shape[1])
            # # y = random.randint(0,image.shape[0])

            # 剪裁image，到指定剪裁尺寸crop_size
            image = safe_crop(image, x, y, crop_size)
            # 剪裁trimap，到指定剪裁尺寸crop_size
            trimap = safe_crop(trimap, x, y, crop_size)

            # 随机翻转
            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                trimap = np.fliplr(trimap)
            """

            #save the image/trimap crop patch
            # patch_save_dir = "show_data_loader_20190330"
            # if not os.path.exists(patch_save_dir):
            #     os.makedirs(patch_save_dir)
            # image_patch_path = patch_save_dir + '/' + img_name_prefix + '_image_' + str(i_batch) + '.png'
            # trimap_patch_path = patch_save_dir + '/' + img_name_prefix + '_trimap_' + str(i_batch) + '.png'
            # cv2.imwrite(image_patch_path,image)
            # cv2.imwrite(trimap_patch_path,trimap)

            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

            batch_x[i_batch] = image/255.0
            batch_y[i_batch] = make_mask_for_batch_y(mask) 

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)

class ValDataGen(Sequence):
    def __init__(self, usage):
        self.usage = usage
        # filename会是"train_names.txt"或"valid_names.txt"
        # "train_names.txt"、"valid_names.txt"中图片名，例如"0-100.png"
        filename = '{}_names.txt'.format(usage)
        with open(filename, 'r') as f:
            self.names = f.read().splitlines()
        np.random.shuffle(self.names)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        # 每一个epoch，从前往后读取self.ids，依据id，读取self.names
        # idx应为第几个batch，i为该次batch的起始点
        i = idx * batch_size
        # length为当前batch的大小
        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 3), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols, num_classes), dtype=np.uint8)

        for i_batch in range(length):
            ###normal
            img_name = self.names[i] # xx.jpg
            image_path = os.path.join(rgb_image_path, img_name)
            image = cv2.imread(image_path,1)

            img_name_prefix = img_name.split('.')[0]
            mask_name = img_name_prefix+".png"
            mask_path = os.path.join(mask_img_path, mask_name)
            mask = cv2.imread(mask_path,0)

            image = cv2.resize(image, dsize=(img_rows, img_cols), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=(img_rows, img_cols), interpolation=cv2.INTER_NEAREST)

            """
            image = pad_and_resize_to_target_size(image,img_cols,img_rows)
            mask = pad_and_resize_mask_to_target_size(mask,img_cols,img_rows)
            trimap = generate_random_trimap(mask)

            #save the image/trimap crop patch
            patch_save_dir = "show_data_loader_val_20190330"
            if not os.path.exists(patch_save_dir):
                os.makedirs(patch_save_dir)
            image_patch_path = patch_save_dir + '/' + img_name_prefix + '_image_' + str(i_batch) + '.png'
            trimap_patch_path = patch_save_dir + '/' + img_name_prefix + '_trimap_' + str(i_batch) + '.png'
            cv2.imwrite(image_patch_path,image)
            cv2.imwrite(trimap_patch_path,trimap)
            
            batch_x[i_batch] = image/255.0
            batch_y[i_batch] = make_trimap_for_batch_y(trimap) 
            """

            batch_x[i_batch] = image/255.0
            batch_y[i_batch] = make_mask_for_batch_y(mask) 
            
            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)

def train_gen():
    return DataGenSequence('train')

def valid_gen():
    return ValDataGen('valid')

if __name__ == '__main__':
    pass
