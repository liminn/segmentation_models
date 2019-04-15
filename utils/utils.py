import multiprocessing
import os
import random
import math

import cv2
import keras.backend as K
from keras.utils import to_categorical

import numpy as np
from tensorflow.python.client import device_lib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

def set_npy_weights(weights_path, model):
    npy_weights_path = os.path.join("./segmentation_models/pspnet_temp/pretrained_weights", "npy", weights_path + ".npy")
    print(npy_weights_path)
    json_path = os.path.join("./segmentation_models/pspnet_temp/pretrained_weights", "keras", weights_path + ".json")
    print(json_path)
    h5_path = os.path.join("./segmentation_models/pspnet_temp/pretrained_weights", "keras", weights_path + ".h5")
    print(h5_path)

    print("Importing weights from %s" % npy_weights_path)
    weights = np.load(npy_weights_path,encoding="latin1").item()

    for layer in model.layers:
        print(layer.name)
        if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
            mean = weights[layer.name]['mean'].reshape(-1)
            variance = weights[layer.name]['variance'].reshape(-1)
            scale = weights[layer.name]['scale'].reshape(-1)
            offset = weights[layer.name]['offset'].reshape(-1)
            model.get_layer(layer.name).set_weights([scale, offset, mean, variance])
        elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
            try:
                weight = weights[layer.name]['weights']
                model.get_layer(layer.name).set_weights([weight])
            except Exception as err:
                try:
                    biases = weights[layer.name]['biases']
                    model.get_layer(layer.name).set_weights([weight,
                                                             biases])
                except Exception as err2:
                    print(err2)
        if layer.name == 'activation_52':
            break

# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()

# Plot the training and validation loss + accuracy
def plot_training(history,pic_name='train_val_loss.png'):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.history['loss'], label="train_loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.plot(history.history['acc'],label="train_acc")
    plt.plot(history.history['val_acc'],label="val_acc")
    plt.title("Train/Val Loss and Train/Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Acc")
    plt.legend(loc="upper left")
    plt.savefig(pic_name)

def random_rescale_image_and_mask(image,mask,min_scale = 0.5, max_scale = 2):
    rows = image.shape[0]
    cols = image.shape[1]
    # print("image.shape:{}".format(image.shape))
    # print("mask.shape:{}".format(mask.shape))
    # print("rows:{},cols:{}".format(rows,cols))
    ratio = random.uniform(min_scale,max_scale)
    # print("ratio:{}".format(ratio))
    new_rows = int(ratio*rows)
    new_cols = int(ratio*cols)
    # print("new_rows:{},new_cols:{}".format(new_rows,new_cols))
    image = cv2.resize(image, dsize=(new_cols, new_rows), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, dsize=(new_cols, new_rows), interpolation=cv2.INTER_LINEAR)
    # print("image.shape:{}".format(image.shape))
    # print("mask.shape:{}".format(mask.shape))
    return image,mask

def generate_random_trimap(alpha):
    # ### 非0区域置为255，然后膨胀及收缩，多出的部分为128区域
    # ### 优点：如果有一小撮头发为小于255，但大于0的，那通过该方法，128区域会覆盖到该一小撮头发部分
    # mask = alpha.copy()                             # 0~255
    # # 非纯背景置为255
    # mask = ((mask!=0)*255).astype(np.float32)       # 0.0和255.0
  
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # # 如果尺寸过小(总面积小于500*500)，则减半膨胀和腐蚀的程度
    # if(alpha.shape[0]* alpha.shape[1] < 250000):
    #     dilate = cv2.dilate(mask, kernel, iterations=np.random.randint(5, 7))   # 膨胀少点
    #     erode = cv2.erode(mask, kernel, iterations=np.random.randint(7, 10))    # 腐蚀多点
    # else:
    #     dilate = cv2.dilate(mask, kernel, iterations=np.random.randint(10, 15)) # 膨胀少点
    #     erode = cv2.erode(mask, kernel, iterations=np.random.randint(15, 20))   # 腐蚀多点

    # ### for循环生成trimap，特别慢
    # # for row in range(mask.shape[0]):
    # #     for col in range(mask.shape[1]):
    # #         # 背景区域为第0类
    # #         if(dilate[row,col]==255 and mask[row,col]==0):
    # #             img_trimap[row,col]=128
    # #         # 前景区域为第1类
    # #         if(mask[row,col]==255 and erode[row,col]==0):
    # #             img_trimap[row,col]=128

    # ### 操作矩阵生成trimap，特别快
    # # ((mask-erode)==255.0)*128  腐蚀掉的区域置为128
    # # ((dilate-mask)==255.0)*128 膨胀出的区域置为128
    # # + erode 整张图变为255/0/128
    # img_trimap = ((mask-erode)==255.0)*128 + ((dilate-mask)==255.0)*128 + erode

    # return img_trimap.astype(np.uint8)
    
    mask = alpha.copy()                                                         # 0~255
    # 非纯背景置为255
    mask = ((mask!=0)*255).astype(np.float32)                                   # 0.0和255.0
    #mask = ((mask==255)*255).astype(np.float32)                                # 0.0和255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate = cv2.dilate(mask, kernel, iterations=np.random.randint(1, 5)) 
    erode = cv2.erode(mask, kernel, iterations=np.random.randint(1, 5))   
    # 128/255/0
    img_trimap = ((mask-erode)==255.0)*128 + ((dilate-mask)==255.0)*128 + erode
    # 加上本来是128的区域
    #bool_unkonw = (alpha!=255)*(alpha!=0)
    #img_trimap = img_trimap*(1-bool_unkonw)+bool_unkonw*128
    return img_trimap.astype(np.uint8)

# Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
def random_choice(trimap, crop_size=(512, 512)):
    crop_height, crop_width = crop_size
    # np.where(arry)：输出arry中‘真’值的坐标(‘真’也可以理解为非零)
    # 返回：(array([]),array([])) 第一个array([])是行坐标，第二个array([])是列坐标
    y_indices, x_indices = np.where(trimap == 128)
    # 未知像素的数量
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        # 任取一个未知像素的坐标
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        # 为下面的剪裁提供起始点
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
    return x, y

def safe_crop(mat, x, y, crop_size=(512, 512)):
    # 例如：crop_height = 640，crop_width = 640
    crop_height, crop_width = crop_size
    # 对于alpha，先建立尺寸为(crop_height, crop_width)的全0数组
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    # 对于fg,bg,image，先建立尺寸为(crop_height, crop_width,3)的全0数组
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    # 注意：这里是函数名为safe_crop的原因！
    # 若(y+crop_height)超出了mat的范围，则也不会报错，直接取到mat的边界即停止
    # 因此crop的尺寸不一定是(crop_height,crop_height)，有可能小于(crop_height,crop_height)
    crop = mat[y:y+crop_height, x:x+crop_width]
    # 得到crop的尺寸
    h, w = crop.shape[:2]
    # 将crop所包含的内容，赋给ret
    # 当然，ret其余部分为0
    ret[0:h, 0:w] = crop
    # # 缩放到(img_rows,img_cols)，即(320,320)
    # if crop_size != (img_rows, img_cols):
    #     # dsize即指的是Size(width，height)
    #     print("crop_size != (512,512)")
    #     ret = cv2.resize(ret, dsize=(img_rows, img_cols), interpolation=cv2.INTER_NEAREST)

    return ret

def make_trimap_for_batch_y(trimap):
    for row in range(trimap.shape[0]):
        for col in range(trimap.shape[1]):
            # 背景区域为第0类
            if(trimap[row,col]==0):
                trimap[row,col]=0
            # 前景区域为第1类
            if(trimap[row,col]==255):
                trimap[row,col]=1
            # 128区域为第2类
            if(trimap[row,col]==128): 
                trimap[row,col]=2
    trimap = to_categorical(trimap, 3) 
    return trimap

def make_mask_for_batch_y(mask):
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            # 背景区域为第0类
            if(mask[row,col]==0):
                mask[row,col]=0
            # 前景区域为第1类
            if(mask[row,col]==255):
                mask[row,col]=1
    mask = to_categorical(mask, 2) 
    return mask

def colorful(out):
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

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def pad_and_resize_to_target_size(image, target_rows, target_cols,interpolation = cv2.INTER_LINEAR):
    rows,cols = image.shape[0],image.shape[1]
    if rows>cols:
        mat = np.zeros((rows, rows, 3), np.float32)
    else:
        mat = np.zeros((cols, cols, 3), np.float32)
    mat[0:rows, 0:cols,:] = image
    mat = cv2.resize(mat,(target_cols,target_rows),interpolation=interpolation)
    return mat.astype(np.uint8)

def pad_and_resize_mask_to_target_size(image, target_rows, target_cols,interpolation = cv2.INTER_NEAREST):
    rows,cols = image.shape[0],image.shape[1]
    if rows>cols:
        mat = np.zeros((rows, rows), np.float32)
    else:
        mat = np.zeros((cols, cols), np.float32)
    mat[0:rows, 0:cols] = image
    mat = cv2.resize(mat,(target_cols,target_rows),interpolation=interpolation)
    return mat.astype(np.uint8)

def vis_segmentation(image, seg_map, save_path_name = "examples.png"):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 6])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = seg_map
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  # ax = plt.subplot(grid_spec[3])
  # legend_elements = [Line2D([0], [0], color='black', lw=4, label='Background'),
  #                    Line2D([0], [0], color='gray', lw=4, label='Unknow Area'),
  #                    Line2D([0], [0], color='white', lw=4, label='Foreground')]
  # ax.legend(handles=legend_elements,loc = "center")
  # plt.axis('off')
  # plt.show()

  plt.savefig(save_path_name)
  plt.close('all')
  
if __name__ == '__main__':
    ### test generator_random_trimap()
    # img_mask = cv.imread('trimap_test_out/supervisely4847.png',0) 
    # i = 0
    # for i in list(range(10)):
    #     generator_random_trimap(img_mask,i)

    ### test random_rescale_image_and_mask()
    image  = cv2.imread("./temp/image/supervisely5641.jpg",1)
    mask  = cv2.imread("./temp/mask/supervisely5641.png",0)
    image,mask = random_rescale_image_and_mask(image,mask)
    cv2.imwrite("./temp/image/image_new.png",image)
    cv2.imwrite("./temp/mask/mask_new.png",mask)




