from mrcnn import model as modellib, utils

import requests
import zipfile
import io
import os

# Download balloon dataset if required
from samples.balloon.balloon import BalloonConfig, train

ZIP_FILE_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip"

r = requests.get(ZIP_FILE_URL)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("/storage")

#python3 balloon.py train --dataset=/path/to/dataset --model=coco


weights = "coco"
dataset = "/storage/balloon"
logs = '/artifacts'

COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5"

# Configurations

config = BalloonConfig()
config.display()

# Create model
model = modellib.MaskRCNN(mode="training", config=config, model_dir=logs)

# Select weights file to load
if weights.lower() == "coco":
    weights_path = COCO_WEIGHTS_PATH
    # Download weights file
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)
elif weights.lower() == "last":
    # Find last trained weights
    weights_path = model.find_last()
elif weights.lower() == "imagenet":
    # Start from ImageNet trained weights
    weights_path = model.get_imagenet_weights()
else:
    weights_path = weights

# Load weights
print("Loading weights ", weights_path)
if weights.lower() == "coco":
    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
else:
    model.load_weights(weights_path, by_name=True)

# Train or evaluate
train(model, dataset, learning_rate=config.LEARNING_RATE)


