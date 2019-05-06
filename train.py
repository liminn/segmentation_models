import os
import yaml

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,TensorBoard,CSVLogger
from keras.utils import multi_gpu_model
from keras.optimizers import Adam

# 待改
from segmentation_models.pspnet_temp.pspnet_model import build_pspnet
from segmentation_models import FPN, PSPNet
from segmentation_models.metrics import iou_score
from segmentation_models.losses import bce_jaccard_loss,jaccard_loss

from utils.customize import CustomizeModelCheckpoint
from utils.utils import get_available_cpus, get_available_gpus, plot_training,set_npy_weights,get_txt_length

from data_generator import train_gen, valid_gen

if __name__ == '__main__':
    
    # Set specific config file
    config_path = "configs/fpn.yaml"
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read())
        #print(cfg.items())

    # Use specific GPU
    if cfg["TRAINNING"]["SPECIFIC_GPU_NUM"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["TRAINNING"]["SPECIFIC_GPU_NUM"])

    # Set model save path
    checkpoint_models_path = cfg["CHECKPOINT"]["MODEL_DIR_BASE"] + '/' +  cfg["CHECKPOINT"]["MODEL_DIR"]
    if not os.path.exists(checkpoint_models_path):
        os.makedirs(checkpoint_models_path)
    
    # Callbacks
    log_dir = './logs/' + cfg["CHECKPOINT"]["MODEL_DIR"]
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    model_save_path = checkpoint_models_path +'/'+'model-{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=False)
    #early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.8, patience=cfg["TRAINNING"]["PATIENCE"], verbose=1, min_lr=1e-8)
    csv_log = CSVLogger('{}/history.log'.format(checkpoint_models_path))

    # Define model
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

    # Use specific GPU or multi GPUs
    if cfg["TRAINNING"]["SPECIFIC_GPU_NUM"] is not None:
        final = model
    else:
        # Multi-GPUs
        num_gpu = len(get_available_gpus())
        if num_gpu >= 2:
            final = multi_gpu_model(model, gpus=num_gpu)
            # rewrite the callback: saving through the original model and not the multi-gpu model.
            model_checkpoint = CustomizeModelCheckpoint(model,model_save_path, monitor='val_loss', verbose=1, save_best_only=True)
        else:
            final = model

    # Finetune the whole network together.
    for layer in final.layers:
        layer.trainable = True

    # Final callbacks
    #callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]
    callbacks = [tensor_board, model_checkpoint, reduce_lr, csv_log]

    # Optimizer
    Adam = Adam(lr=cfg["TRAINNING"]["INITIAL_LR"])

    # Compile
    if(cfg["MODEL"]["NUM_CLASSES"]==2):
        loss = 'binary_crossentropy'
        #loss = bce_jaccard_loss
        print("loss:{}".format(loss))
    else:
        loss = 'categorical_crossentropy'
        #loss = jaccard_loss
        print("loss:{}".format(loss))
    final.compile(optimizer=Adam, loss = loss, metrics=[iou_score]) 
    final.summary()

    # Start trainning
    num_cpu = get_available_cpus()
    workers = int(round(num_cpu / 4))
    nums_train = get_txt_length(cfg["DATA"]["TRAIN_TXT_PATH"])
    print("nums_train:{}".format(nums_train))
    nums_valid = get_txt_length(cfg["DATA"]["VALID_TXT_PATH"])
    print("nums_valid:{}".format(nums_valid))
    final.fit_generator(
                        generator = train_gen(),
                        steps_per_epoch = nums_train // cfg["TRAINNING"]["BATCH_SIZE"],
                        #steps_per_epoch = 10, # for test
                        validation_data = valid_gen(),
                        validation_steps = nums_valid // cfg["TRAINNING"]["BATCH_SIZE"],
                        epochs = cfg["TRAINNING"]["EPOCHS"],
                        verbose = 1,
                        callbacks = callbacks,
                        use_multiprocessing = True,
                        workers = workers
                        )
