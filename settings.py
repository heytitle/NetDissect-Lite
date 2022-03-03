import os
import numpy as np
import json

######### global settings  #########
GPU = True                                  # running on GPU is highly suggested
TEST_MODE = False                           # turning on the testmode means the code will run on a small dataset.
CLEAN = True                               # set to "True" if you want to clean the temporary large files after generating result
VENDOR = ["torchvision", "netdissect"][1]  # model vendor: {torchvision, netdissect}
MODEL = 'vgg16'                            # model arch: resnet18, alexnet, resnet50, densenet161
DATASET = ["imagenet", "places365"][1]                       # model trained on: places365 or imagenet
QUANTILE = 0.005                            # the threshold used for activation
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
SCORE_THRESHOLD = 0.04                      # the threshold used for IoU score (in HTML file)
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 1                                # how many process is used for tallying (Experiments show that 1 is the fastest)
CATAGORIES = ["object", "part","scene","texture","color"] # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"

# Pat: This allows us to use the (fast) scatch file system on the clusters
FEATURE_NAME = ["conv4_3", "conv5_3"][1]
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "./result")
OUTPUT_FOLDER = f"{ARTIFACT_DIR}/{VENDOR}-{MODEL}-{DATASET}"

print(f"Output dir: {OUTPUT_FOLDER}")

########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

if MODEL != 'alexnet':
    DATA_DIRECTORY = 'dataset/broden1_224'
    IMG_SIZE = 224
else:
    DATA_DIRECTORY = 'dataset/broden1_227'
    IMG_SIZE = 227

if DATASET == 'places365':
    NUM_CLASSES = 365
elif DATASET == 'imagenet':
    NUM_CLASSES = 1000
if MODEL == 'resnet18':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
        MODEL_PARALLEL = True
    elif DATASET == 'imagenet':
        MODEL_FILE = None
        MODEL_PARALLEL = False
elif MODEL == 'densenet161':
    FEATURE_NAMES = ['features']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_densenet161_places365_python36.pth.tar'
        MODEL_PARALLEL = False
elif MODEL == 'resnet50':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_resnet50_places365_python36.pth.tar'
        MODEL_PARALLEL = False
elif MODEL == 'vgg16':
    if VENDOR == "torchvision":
        # We use pretrained then
        MODEL_FILE = os.path.expanduser("~/.cache/torch/hub/checkpoints/vgg16-397923af.pth")

        # Remark: to keep minimal change, we follow their color formating
        # - The mean is in `Blue Green Red (BGR)` format.
        #   See 1) https://github.com/CSAILVision/NetDissect-Lite/blob/master/feature_operation.py#L27
        #       2) https://github.com/CSAILVision/NetDissect-Lite/blob/master/feature_operation.py#L57
        # - The std is in `RGB` format
        #   See https://github.com/CSAILVision/NetDissect-Lite/blob/master/feature_operation.py#L63

        # How does tihs mix format actually come in to play?
        # 1. normalize_image(..., bgr=..) loads the image (shape: [h, w, rgb]) normally and reverse the channel dimensions [h, w, bgr] before using `the BGR mean`
        #   See https://github.com/CSAILVision/NetDissect-Lite/blob/2163454ebeb5b15aac64e5cbd4ed8876c5c200df/loader/data_loader.py#L680
        # 2. The function then swap dimensions into [bgr, h, w]
        # 3. Before feeding to model, the color dimension is reversed again into [rgb, h, w] and devide by the RGB std
        #   See https://github.com/CSAILVision/NetDissect-Lite/blob/master/feature_operation.py#L62

        # Ref: https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/data/constants.py#L2
        STANDARD_RGB_IMAGENET_MEAN = [0.485, 0.456, 0.406]

        # This is the unit of pixel values
        NORMALIZATION_BGR_MEAN = (255 * np.array(STANDARD_RGB_IMAGENET_MEAN[::-1])).tolist()

        # This is the unit of [0, 1] values (normaized pixel values)
        # Ref: https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/data/constants.py#L2
        NORMALIZATION_RGB_STD = [0.229, 0.224, 0.225]

    elif VENDOR == "netdissect":
        # This is the unit of pixel values
        # Ref: https://github.com/CSAILVision/NetDissect-Lite/blob/master/feature_operation.py#L27
        NORMALIZATION_BGR_MEAN = [109.5388, 118.6897, 124.6901]

        # This is the unit of [0, 1] values (normaized pixel values)
        NORMALIZATION_RGB_STD = [1., 1., 1.]
        if DATASET == "imagenet":
            MODEL_FILE = "zoo/netdissect-vgg16_imagenet-2b51436b.pth"
        elif DATASET == "places365":
            MODEL_FILE = "zoo/netdissect-vgg16_places365-dab93d8c.pth"
    else:
        raise ValueError(f"`{VENDOR}`` is not available.")
    FEATURE_NAMES = [FEATURE_NAME]
    MODEL_PARALLEL = False

print("Using model-file", MODEL_FILE)

if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 1
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = 'index_sm.csv'
    OUTPUT_FOLDER += "_test"
else:
    WORKERS = 12
    BATCH_SIZE = 64
    TALLY_BATCH_SIZE = 64
    TALLY_AHEAD = 4
    INDEX_FILE = 'index.csv'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
with open(f"{OUTPUT_FOLDER}/meta_{FEATURE_NAME}.json", "w") as fh:
    json.dump(dict(model_file=MODEL_FILE), fh, indent=4,  sort_keys=True)