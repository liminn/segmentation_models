### 1.Model
# 待加强，为不同的模型，指定不同的超参数
MODEL_NAME = "PSPNet_temp"
INPUT_ROWS = 512
INPUT_COLS = 512
NUM_CLASSES = 3

### 2.Data
IMAGE_PATH = ""
LABEL_PATH = ""

NUM_TRAIN = 2109       
NUM_VALID = 263    

### 3.Trainning    
SPECIFIC_GPU_NUM = None
INITIAL_LR = 0.01
BATCH_SIZE = 4
PATIENCE = 2
EPOCHS = 200

### 4.Checkpoint
MODEL_DIR_BASE = 'models'
MODEL_DIR = 'xx'

### 5.Inference
inference_output_path_base = "inference_output"
