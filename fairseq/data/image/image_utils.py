import re
from typing import BinaryIO, List, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np
IMAGE_FILE_EXTENSIONS = {".jpg", ".png"}
import sys
if sys.version_info[0] == 2:
    from six.moves.urllib.request import urlretrieve
else:
    from urllib.request import urlretrieve
from PIL import Image, JpegImagePlugin
def parse_path(path: str) -> Tuple[str, List[int]]:
    """Parse data path which is a path to a .jpg/.png file

      Args:
          path (str): the data path to parse

      Returns:
          file_path (str): the file path
    """

    if Path(path).suffix in IMAGE_FILE_EXTENSIONS:
        return path
    else:
        raise Exception("Unknown image suffix")

def resize_aspect_ratio(img, interpolation=cv2.INTER_LINEAR, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    
    ratio = target_size / max(height, width)    

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)


    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    #size_heatmap = (int(target_w/2), int(target_h/2))

    return resized

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img
 

def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img
def reformat_input(image):
    if type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif type(image) == np.ndarray:
        if len(image.shape) == 2: # grayscale
            img_cv_grey = image
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            img_cv_grey = np.squeeze(image)
            img = cv2.cvtColor(img_cv_grey, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3: # BGRscale
            img = image
            img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4: # RGBAscale
            img = image[:,:,:3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif type(image) == JpegImagePlugin.JpegImageFile:
        image_array = np.array(image)
        img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError('Invalid input type. Supporting format = string(file path or url), bytes, numpy array')

    return img, img_cv_grey


def calculate_ratio(width,height):
    '''
    Calculate aspect ratio for normal use case (w>h) and vertical text (h>w)
    '''
    ratio = width/height
    if ratio<1.0:
        ratio = 1./ratio
    return ratio

def compute_ratio_and_resize(img, width, height, model_width, model_height):
    '''
    Calculate ratio and resize correctly for both horizontal text
    and vertical case
    '''
    ratio = width/height
    resized_width = int(model_height*ratio)
    resized_width = min(resized_width, model_width)
    img = cv2.resize(img,(resized_width, model_height),interpolation=Image.ANTIALIAS)

    # resized_height = min(height, model_height)
    # resized_width = min(width, model_width)
    # img = cv2.resize(img,(resized_width, resized_height),interpolation=Image.ANTIALIAS)
    return img


punc_table = {'。':'.', '，':',', '‘':'\'', '’': '\'', '“':'"', '”':'"', '！': '!'}
def norm_punc(string):
    for punc in punc_table:
        string = re.sub(punc, punc_table[punc], string)
    return string