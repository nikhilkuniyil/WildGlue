from models.matching import Matching
from models.superglue import SuperGlue
from models.superpoint import SuperPoint

# from lightglue import LightGlue, DISK, SIFT, ALIKED, DoGHardNet
# from lightglue.utils import rbd

from helper_functions import assign_device_to_model

import torch
from models.utils import frame2tensor
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import collections 
import gc
import copy
from PIL import Image

import xml.etree.ElementTree as ET

config = {
    # Example configuration; replace with actual configuration parameters as needed
    "superpoint": {
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1
    },
    "superglue": {
        "weights": "outdoor",  # or "indoor" depending on your application
        "sinkhorn_iterations": 20,
        "match_threshold": 0.2,
    }
}

device = torch.device('cpu')
matching_model = assign_device_to_model(config, gpu=False)
superpoint_model = SuperPoint(config).to(device)
running_overlap_count = 0


