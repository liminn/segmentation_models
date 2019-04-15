import os
import yaml
import argparse

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,TensorBoard
from keras.utils import multi_gpu_model
from keras.optimizers import Adam

# 待改
from segmentation_models.pspnet_temp.pspnet_model import build_pspnet
from segmentation_models import FPN
from segmentation_models.metrics import iou_score

from utils.customize import CustomizeModelCheckpoint
from utils.utils import get_available_cpus, get_available_gpus, plot_training,set_npy_weights

from data_generator import train_gen, valid_gen

if __name__ == '__main__':

    # Set specific config file
    config_path = "configs/pspnet_temp.yaml"
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read())
        #print(cfg.items())

    # Set model save path
    checkpoint_models_path = cfg["CHECKPOINT"]["MODEL_DIR_BASE"] + '/' +  cfg["CHECKPOINT"]["MODEL_DIR"]
    if not os.path.exists(checkpoint_models_path):
        os.makedirs(checkpoint_models_path)
    
    # Callbacks
    log_dir = './logs/' + cfg["CHECKPOINT"]["MODEL_DIR"]
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    model_save_path = checkpoint_models_path +'/'+'model-{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True)
    #early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.8, patience=cfg["TRAINNING"]["PATIENCE"], verbose=1, min_lr=1e-8)

    # Define model
    if cfg["MODEL"]["MODEL_NAME"] == "PSPNet_temp":
        model = build_pspnet(cfg["MODEL"]["NUM_CLASSES"], resnet_layers=50, input_shape=(cfg["MODEL"]["INPUT_ROWS"],cfg["MODEL"]["INPUT_COLS"]))
        set_npy_weights(weights_path=cfg["MODEL"]["PT_PATH"], model = model)
    else:
        raise Exception("Error: do not support model:{}".format(cfg["MODEL"]["MODEL_NAME"]))

    # Use specific GPU or multi GPUs
    if cfg["TRAINNING"]["SPECIFIC_GPU_NUM"] != None:
        # Use specific GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["TRAINNING"]["SPECIFIC_GPU_NUM"])
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
    callbacks = [tensor_board, model_checkpoint, reduce_lr]

    # Optimizer
    Adam = Adam(lr=cfg["TRAINNING"]["INITIAL_LR"])

    # Compile
    if(cfg["MODEL"]["NUM_CLASSES"]==2):
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    final.compile(optimizer=Adam, loss = loss, metrics=[iou_score]) 
    final.summary()

    # Start fine-tuning
    num_cpu = get_available_cpus()
    workers = int(round(num_cpu / 4))
    #workers = 4
    final.fit_generator(
                        generator = train_gen(),
                        steps_per_epoch = cfg["DATA"]["NUM_TRAIN"] // cfg["TRAINNING"]["BATCH_SIZE"],
                        validation_data = valid_gen(),
                        validation_steps = cfg["DATA"]["NUM_VALID"] //cfg["TRAINNING"]["BATCH_SIZE"],
                        epochs = cfg["TRAINNING"]["EPOCHS"],
                        verbose = 1,
                        callbacks = callbacks,
                        use_multiprocessing = True,
                        workers = workers
                        )
