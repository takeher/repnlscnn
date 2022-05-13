import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import itertools
import colorsys
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import tensorflow as tf
from keras import backend as K
# Root directory of the project
ROOT_DIR = os.path.abspath("./")
DATASET_DIR = "/home/ubuntu/Downloads/cityscapes/"
#datasetfiles = os.listdir(DATASET_DIR+"newImages/")
#datasetfiles = os.listdir(DATASET_DIR+"JPEGImages/")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from config import Config
import utils
import model30 as modellib
weight_path = 'car20220330T1539'
import visualize

import json
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "car_logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


from xml.etree import ElementTree as ET
class devicessConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "car"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024 
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32 ,64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 50

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

config = devicessConfig()
#config.display()

class devicessDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_devicess(self, count, path, mode):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        self.mode = mode
        # Add classes
        self.add_class("car", 1, "car")
        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        image_path = path
        #for i in range(count):
        if mode == "train":
            #2974
            train_path = image_path+"/train/"
            self.datasetfiles = os.listdir(train_path)
            for i in range(count):
                self.add_image("car", image_id=i, path=train_path+self.datasetfiles[i])
        else:
            #1524
            test_path = image_path+"/val/"
            self.datasetfiles = os.listdir(test_path)
            for i in range(count):
                self.add_image("car", image_id=i, path=test_path+self.datasetfiles[i])


    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image = np.array(cv2.imread(info['path']))
        return image
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "car":
            return info["car"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        path = DATASET_DIR+"newLabel/"+self.mode+'/'
        info = self.image_info[image_id]    
        label_path = path+(self.datasetfiles[image_id]).split('.jpg')[0]+"_gtFine_polygons.json"
        #tree = json.loads(label_path)
        with open(label_path, 'r') as j:
            tree = json.loads(j.read())

        weight = tree['imgWidth']
        height = tree['imgHeight']
        cars   = []
        allbox = []
        car_objs = tree["objects"]
        for ob in car_objs:
            cars.append(ob["label"])
            allbox.append(np.array(ob["polygon"]))
        count = len(cars)
        mask  = np.zeros([height, weight, count])
        for i in range(count):
            polygon = allbox[i]
            mask[:,:,i] = cv2.fillPoly(mask[:,:,i].copy(), [polygon], 1)
        class_ids = np.array([self.class_names.index(s) for s in cars])
        return mask, class_ids.astype(np.int32)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, name=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
      if scores[i] > 0.9:
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig(name)
    plt.show()
    if auto_show:
        plt.savefig(name)
        plt.show()
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
datapath = DATASET_DIR+"newImages/"

# Training dataset
dataset_train = devicessDataset()
dataset_train.load_devicess(2900, datapath, "train")
dataset_train.prepare()
# Validation dataset
dataset_val = devicessDataset()
num_val = 100
dataset_val.load_devicess(num_val, datapath, "val")
dataset_val.prepare()
# Load and display random samples
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
# Which weights to start with?
#init_with = "coco"  # imagenet, coco, or last

init_with = None  # imagenet, coco, or last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
elif init_with == "rep":
    # Load the rep model you trained and continue training
    model.load_weights("./mask_rcnn_repvgg.h5", by_name=True)
elif init_with == "res":
    # Load the rep model you trained and continue training
    model.load_weights("./mask_rcnn_resnet.h5", by_name=True)
'''
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE/1,
            epochs=50, 
            layers="all")
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE/1, 
            epochs=150, 
            layers='heads')
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE /10,
            epochs=290, 
            layers="all")
'''
class InferenceConfig(devicessConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#model_path = model.find_last()
model.load_weights("./car_logs/"+weight_path+"/mask_rcnn_car_0290.h5", by_name=True)
# Load trained weights
#print("Loading weights from ", model_path)
#model.load_weights(model_path, by_name=True)
'''
keras_model = model.keras_model
layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers
def get_id_tensor(in_channels):
    input_dim = in_channels
    kernel_value = np.zeros((3, 3, input_dim, in_channels), dtype=np.float32)
    for i in range(in_channels):
        kernel_value[1, 1, i % input_dim, i] = 1
    id_tensor = tf.convert_to_tensor(kernel_value, dtype=np.float32)
    return id_tensor

def deploy_convert_identity(layer, id_tensor):
    kernel = id_tensor
    running_mean = layer.moving_mean
    running_var = layer.moving_variance
    gamma = layer.gamma
    beta = layer.beta
    eps = layer.epsilon
    std = tf.sqrt(running_var + eps)
    t = gamma / std
    return kernel * t, beta - running_mean * gamma / std

def deploy_convert_convbn(conv, bn):
    kernel = conv.get_weights()[0]
    running_mean = bn.moving_mean
    running_var = bn.moving_variance
    gamma = bn.gamma
    beta = bn.beta
    eps = bn.epsilon
    std = tf.sqrt(running_var + eps)
    t = gamma / std
    return kernel * t, beta - running_mean * gamma / std
def pad_1x1_to_3x3_tensor(kernel1x1):
    if kernel1x1 is None:
        return 0
    else:
        return tf.pad(kernel1x1, tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]]))
def get_kernel_bias(conv3, bn3, conv1, bn1, identity=None, id_tensor=None):
    if identity is None:
        kernel3x3, bias3x3 = deploy_convert_convbn(conv3, bn3)
        kernel1x1, bias1x1 = deploy_convert_convbn(conv1, bn1)
        return (
            #kernel3x3 + pad_1x1_to_3x3_tensor(kernel1x1),
            kernel3x3 + kernel1x1,
            bias3x3 + bias1x1,
        )
    else:
        kernel3x3, bias3x3 = deploy_convert_convbn(conv3, bn3)
        kernel1x1, bias1x1 = deploy_convert_convbn(conv1, bn1)
        kernelid, biasid = deploy_convert_identity(identity, id_tensor)
        #return [kernel3x3 + pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid]
        return [kernel3x3 + kernel1x1 + kernelid, bias3x3 + bias1x1 + biasid]
model_deploy = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR,
                          deploy=True)

model_deploy_in = model_deploy.keras_model
layers_deploy = model_deploy.inner_model.layers if hasattr(model_deploy_in, "inner_model") else model_deploy_in.layers


kernels = []
bias    = []
cur_in = [3]+[64]*6+[128]*6+[256]*16
print("#########################")
print("Reparameterization computing")
for i, lay in enumerate(layers):
    if lay.name[-6:] == "x3conv":
        conv3 = layers[i+0]
        conv1 = layers[i+1]
        bn3   = layers[i+2]
        bn1   = layers[i+3]
        if (layers[i+5]).name[-5:]=="ntity":
            identity = layers[i+5]
            id_tensor= get_id_tensor(cur_in[len(kernels)])
        else:
            identity = None
            id_tensor= None
        [kernel, bia] = get_kernel_bias(conv3, bn3, conv1, bn1, identity, id_tensor)
        kernels.append(kernel)
        bias.append(bia)
        #layers_deploy[3*len(kernels)-1].set_weights([K.eval(kernel), K.eval(bia)])
        model_deploy.keras_model.layers[3*len(kernels)-1].set_weights([K.eval(kernel), K.eval(bia)])
    else:
        if i > 274:
            #layers_deploy[i-274+87].set_weights(layers[i].get_weights())
            model_deploy.keras_model.layers[i-274+87].set_weights(layers[i].get_weights())
rep_save_path = r"./mask_rcnn_car_repvgg.h5"

model_deploy.keras_model.save_weights(rep_save_path)
model_deploy.keras_model.load_weights(rep_save_path, by_name=True)
model.load_weights("./car_logs/" + weight_path + "/mask_rcnn_car_0290.h5", by_name=True)
print("save rep model in : "+rep_save_path)
print("########################3")
# Test on a random image
'''
'''
for i in range(10):
    image_id = random.choice(dataset_train.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_train, inference_config, 
                               image_id, use_mini_mask=False)
    results = model.detect([original_image], verbose=1)
    #print(results)
    r = results[0]
    name = "./detectTest/vehicleTest/"+str(i+1)+".jpg"
    display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax(), name=name)
'''
print("detect test")
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
#image_ids = np.random.choice(dataset_val.image_ids, 10)
image_ids = np.arange(num_val)
APs = []
gts = np.zeros(1)
gts_num = []
gts_numlast = 0
preds = np.zeros(1)
preds_num = []
import time
deploy_time = []
model_time = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    #start = time.time()
    #results = model_deploy.detect([image], verbose=0)
    #end = time.time()
    #timeuse_deploy = end - start

    start = time.time()
    results = model.detect([image], verbose=0)
    end = time.time()
    timeuse_model = end - start
    if image_id > 0:
        #deploy_time.append(timeuse_deploy)
        model_time.append(timeuse_model)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps, gt_matchs, pred_matchs =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    if AP>0:
        APs.append(AP)
    else:
        pass
        #APs.append(0)
    gts_numlast = gts_numlast+len(gt_matchs)
    gts_num.append(gts_numlast)
    preds_num.append(len(pred_matchs))
    gts = np.append(gts, gt_matchs)
    preds= np.append(preds, pred_matchs)
#print("rep deploy use time(s):", np.sum(np.array(deploy_time)))
print("rep model  use time(s):", np.sum(np.array(model_time)))
gts = np.delete(gts,0)
gt_per = []
for i, nums in enumerate(preds_num):
    gt_per = gt_per+[gts_num[i]]*nums
preds = np.delete(preds,0)
# Compute precision and recall at each prediction box step
precisions = abs(np.sort(-np.cumsum(preds > -1) / (np.arange(len(preds)) + 1)))
recalls = np.sort(np.cumsum(preds > -1).astype(np.float32) / gt_per)
# Pad with start and end values to simplify the math
precisions = np.concatenate([[1], precisions, [0]])
recalls = np.concatenate([[0], recalls, [1]])
print(len(APs))
print(APs)
print("mAP: ", np.mean(APs))
plt.plot(recalls, precisions)
plt.title("P-R curve")
plt.xlabel("Recalls")
plt.ylabel("Precisions")
plt.show()
