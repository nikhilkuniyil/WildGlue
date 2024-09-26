import torch
import matplotlib.pyplot as plt
import cv2
import numpy as  np

import sys
sys.path.append('/home/nkuniyil/nkuniyil/SuperGluePretrainedNetwork/padded-transformations')
import padtransf
from padtransf import warpPerspectivePadded

from models.matching import Matching
from models.superglue import SuperGlue
from models.superpoint import SuperPoint
from models.utils import frame2tensor
import xml.etree.ElementTree as ET
import collections


import gc
import copy

def plot_image_pair(path1):
    img1 = cv2.imread(path1)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    plt.imshow(img1_gray)
    plt.show()

def spsg_model(config):
    matching_model = Matching(config)
    return matching_model

def assign_device_to_model(config, gpu=False):
    # Check if CUDA is available
    if torch.cuda.is_available() and gpu:
        device = torch.device('cuda')  # Use the default CUDA device
    else:
        device = torch.device('cpu')
    model = spsg_model(config)
    model.to(device)
    return model

def format_image_path(image_num):
    if image_num < 10:
        img_path = f'images/EnrNE_Day2_Run1__000{image_num}.jpg'
    elif image_num < 100:
        img_path = f'images/EnrNE_Day2_Run1__00{image_num}.jpg'
    else:
        img_path = f'images/EnrNE_Day2_Run1__0{image_num}.jpg'
    return img_path

def downsample_img_pair(scale_factor, img1, img2):
    resized_img1 = cv2.resize(img1, (0, 0), fx=scale_factor, fy=scale_factor)
    resized_img2 = cv2.resize(img2, (0, 0), fx=scale_factor, fy=scale_factor)
    return resized_img1, resized_img2

def plot_stitched_image(img1, img2, scale_factor):
    img_left, img_right = downsample_img_pair(scale_factor, img1, img2)
    dst_padded, src_warped, shifted_transf = warpPerspectivePadded(img_left, img_right, H)

    alpha = 0.5
    beta = 1 - alpha
    blended = cv2.addWeighted(src_warped, alpha, dst_padded, beta, 1.0)

    plt.imshow(blended)
    plt.show()

def match_descriptors(descriptors1, descriptors2, k_value):
    if (descriptors1 is None) or (len(descriptors1) == 0) or (descriptors2 is None) or (len(descriptors2) == 0):
        return False
     # Initialize the matcher
    bf = cv2.BFMatcher()

    # Match descriptors (find the best match for each descriptor)
    matches = bf.knnMatch(descriptors1, descriptors2, k=k_value)
    return matches


# utilize lowe's ratio test to filter matches
def filter_keypoint_matches(matches, ratio=0.7):
    matches = sorted(matches, key=lambda x: x[0].distance)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches
    

def SIFT_matching_keypoints(img1, img2):
    # detect key points and descriptors for each image
    sift = cv2.SIFT_create(nfeatures=5000)

    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Match descriptors (find the best match for each descriptor)
    matches = match_descriptors(descriptors1, descriptors2, k_value=2)
    if isinstance(matches, bool):
        return False

    # Apply ratio test to filter the matches
    good_matches = filter_keypoint_matches(matches, ratio=0.75)
            
    # Convert keypoints to numpy arrays
    matching_points1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matching_points2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return matching_points1, matching_points2


def ORB_matching_keypoints(img1, img2):
    orb = cv2.ORB_create()

    # compute the descriptors with ORB
    keypoints1, descriptors1 = orb.compute(img1, None)
    keypoints2, descriptors2 = orb.compute(img2, None)

    # match descriptors
    matches = match_descriptors(descriptors1, descriptors2, k_value=2)
    if isinstance(matches, bool):
        return False

    # filter the matches based on ratio
    good_matches = filter_keypoint_matches(matches, ratio=0.75)

    # Convert keypoints to numpy arrays
    matching_points1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matching_points2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return matching_points1, matching_points2


def BRISK_matching_keypoints(img1, img2):
    # Initiate BRISK descriptor
    BRISK = cv2.BRISK_create()

    # Find the keypoints and compute the descriptors for input and training-set image
    keypoints1, descriptors1 = BRISK.detectAndCompute(img1, None)
    keypoints2, descriptors2 = BRISK.detectAndCompute(img2, None)

    # match descriptors
    matches = match_descriptors(descriptors1, descriptors2, k_value=2)
    if isinstance(matches, bool):
        return False

    # filter the matches based on ratio
    good_matches = filter_keypoint_matches(matches, ratio=0.75)

    # Convert keypoints to numpy arrays
    matching_points1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matching_points2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return matching_points1, matching_points2


def KAZE_matching_keypoints(img1, img2):
    # Initiate KAZE descriptor
    KAZE = cv2.KAZE_create()

    # Find the keypoints and compute the descriptors for input and training-set image
    keypoints1, descriptors1 = KAZE.detectAndCompute(img1, None)
    keypoints2, descriptors2 = KAZE.detectAndCompute(img2, None)

    # match descriptors
    matches = match_descriptors(descriptors1, descriptors2, k_value=2)
    if isinstance(matches, bool):
        return False

    # filter the matches based on ratio
    good_matches = filter_keypoint_matches(matches, ratio=0.75)

    # Convert keypoints to numpy arrays
    matching_points1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matching_points2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return matching_points1, matching_points2


def BRIEF_matching_keypoints(img1, img2):
    # Initiate KAZE descriptor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # Compute the BRIEF descriptor for the keypoints
    keypoints1, descriptors1 = brief.compute(img1, keypoints1)
    keypoints2, descriptors2 = brief.compute(img2, keypoints2)

    # match descriptors
    matches = match_descriptors(descriptors1, descriptors2, k_value=2)

    # ensure that each image has a respective descriptor
    if isinstance(matches, bool):
        return False
    
    # filter the matches based on ratio
    good_matches = filter_keypoint_matches(matches, ratio=0.75)

    # Convert keypoints to numpy arrays
    matching_points1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matching_points2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return matching_points1, matching_points2


def extract_matching_indices(model_output):
    # extract original key points
    keypoints0, keypoints1 = model_output['keypoints0'], model_output['keypoints1']

    # convert to numpy array
    keypoints0_np = keypoints0[0].cpu().numpy()
    keypoints1_np = keypoints1[0].cpu().numpy()

    # extract matching indices
    matches0 = model_output['matches0']
    matches0_np = matches0[0].cpu().numpy()

    valid_matches = matches0_np != -1
    valid_matches0_indices = np.arange(matches0_np.shape[0])[valid_matches]
    valid_matches1_indices = matches0_np[valid_matches]

    match_keypoints0 = keypoints0_np[valid_matches0_indices, :]
    match_keypoints1 = keypoints1_np[valid_matches1_indices, :]

    return match_keypoints0, match_keypoints1

# remove indexing 0 for SPSG, add it for SIFT
def points_to_set(points):
    flattened_points = [tuple(point) for point in points]
    return set(flattened_points)


def filter_with_ransac(match_keypoints0, match_keypoints1): 
    # check whether there are enough matches to compute a homography
    if len(points_to_set(match_keypoints0)) < 10 or len(points_to_set(match_keypoints1)) < 10:
        return False
    
    points1 = np.float32([kp for kp in match_keypoints0])
    points2 = np.float32([kp for kp in match_keypoints1])


    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 1.0)
    return H

def convert_to_list_of_lists(large_list):
    '''Converts weirdly formatted list to list of lists'''
    for arr in large_list:
        list_of_lists = [arr.tolist() for arr in large_list]
    return list_of_lists

def draw_points(points_list, img):
    '''Draws keypoints on image'''
    for point in points_list:
        int_point = [point[0], point[1]]
        img = cv2.circle(img, int_point, 10, (0, 0, 255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def transform_points_to_original(points, scale_factor):
    scale_factor = 1 / scale_factor
    return [(x * scale_factor, y * scale_factor) for x,y in points]


def scale_points(scale_factor, points):
    points = np.array(points)
    points = points * scale_factor
    return list(points)


def convert_box_coordinates(xtl, ytl, xbr, ybr, shifted_transf):
    tl_coords = [xtl, ytl, 1]
    br_coords = [xbr, ybr, 1]

    transformed_tl_homg = np.dot(shifted_transf, tl_coords)
    transformed_br_homg = np.dot(shifted_transf, br_coords)

    transformed_tl = transformed_tl_homg[:2] / transformed_tl_homg[2]
    transformed_br = transformed_br_homg[:2] / transformed_br_homg[2]

    return transformed_tl[0], transformed_tl[1], transformed_br[0], transformed_br[1]

# returns true if the two boxes overlap
def overlap(source, target):
    # unpack points
    tl1, br1 = source[:2], source[2:]
    tl2, br2 = target[:2], target[2:]

    # checks
    if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
        return False
    if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
        return False
    return True

def generateOffsets(src, M, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0):
    assert M.shape == (3, 3), \
        'Perspective transformation shape should be (3, 3).\n' \
        + 'Use warpAffinePadded() for (2, 3) affine transformations.'

    M = M / M[2, 2]  # Normalize M to ensure a legal homography
    if flags in (cv2.WARP_INVERSE_MAP, cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP):
        M = cv2.invert(M)[1]  # Invert M if necessary
        flags -= cv2.WARP_INVERSE_MAP

    src_h, src_w = src.shape[:2]
    lin_homg_pts = np.array([[0, src_w, src_w, 0], [0, 0, src_h, src_h], [1, 1, 1, 1]])

    transf_lin_homg_pts = M.dot(lin_homg_pts)
    transf_lin_homg_pts /= transf_lin_homg_pts[2, :]

    min_x = np.floor(np.min(transf_lin_homg_pts[0])).astype(int)
    min_y = np.floor(np.min(transf_lin_homg_pts[1])).astype(int)

    # Determine padding offsets
    anchor_x, anchor_y = 0, 0
    if min_x < 0:
        anchor_x = -min_x
    if min_y < 0:
        anchor_y = -min_y

    return anchor_x, anchor_y

def pixel_distance(list1, list2):
    dist = 0
    for i in range(len(list1)):
        dist += (list1[i] - list2[i]) ** 2

    if np.sqrt(dist) <= 60:
        return True
    return False

def getOverlaps(boxes_list1, boxes_list2):
    overlapCount = 0
    for i in range(len(boxes_list1)):
        for j in range(len(boxes_list2)):
            if overlap(boxes_list1[i], boxes_list2[j]) or pixel_distance(boxes_list1[i], boxes_list2[j]):
                overlapCount += 1  
                break
            
    return overlapCount

def adjust_bounding_box_for_padding(box, padding_offsets):
    tl, br = box[:2], box[2:]  # Top-left and bottom-right corners
    # Adjust coordinates by padding offsets
    tl_adjusted = [tl[0] + padding_offsets[0][0], tl[1] + padding_offsets[0][1]]
    br_adjusted = [br[0] + padding_offsets[0][0], br[1] + padding_offsets[0][1]]
    return [tl_adjusted[0], tl_adjusted[1], br_adjusted[0], br_adjusted[1]]


def overlap_per_img_pair(img1_coords, img2_coords, scale_factor, padding_offsets, shifted_transf):
    # check if neither image has boxes
    if img1_coords == [] or img2_coords == []:
        return 0
    
    # 1st image
    for i in range(len(img1_coords)):
        img1_coords[i] = scale_points(scale_factor, img1_coords[i])
    
    transformed_points_1 = []
    for box in img1_coords:
        xtl, ytl, xbr, ybr = box
        trans_xtl, trans_ytl, trans_xbr, trans_ybr = convert_box_coordinates(xtl, ytl, xbr, ybr, shifted_transf)
        transformed_points_1.append([trans_xtl, trans_ytl, trans_xbr, trans_ybr])
    
    # 2nd image
    for i in range(len(img2_coords)):
        img2_coords[i] = scale_points(scale_factor, img2_coords[i])
    
    new_coords_2 = []
    for box in img2_coords:
        trans_xtl, trans_ytl, trans_xbr, trans_ybr = adjust_bounding_box_for_padding(box, padding_offsets)
        new_coords_2.append([trans_xtl, trans_ytl, trans_xbr, trans_ybr])
    
    # find total number of bounding boxes that overlap
    overlapCount = getOverlaps(transformed_points_1, new_coords_2)
    return overlapCount

def convert_to_tuples(array_list):
    """ Convert a list of numpy arrays to a list of tuples. """
    return [tuple(arr) for arr in array_list]
 
def check_membership(sift_points1, sift_points2, spsg_points1, matching_points0, matching_points1):
    # convert to sets for fast lookup
    spsg_set_list1 = set(convert_to_tuples(spsg_points1))

    for i in range(len(sift_points1)):
        if list(sift_points1)[i] not in spsg_set_list1:
            matching_points0.append(list(sift_points1)[i])
            matching_points1.append(list(sift_points2)[i])
    return matching_points0, matching_points1

def remove_corresponding_duplicates(list1, list2):
    set1, set2 = set(), set()
    new_list1, new_list2 = [], []

    for i in range(len(list1)):
        if list1[i] not in set1 and list2[i] not in set2:
            new_list1.append(list1[i])
            new_list2.append(list2[i])

            set1.add(list1[i])
            set2.add(list2[i])
    return new_list1, new_list2 

def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def clear_memory():
    del image1_tensor, image2_tensor, down_img1, down_img2
    gc.collect()
    torch.cuda.empty_cache()    

def convert_annotations_to_dict(annotations_file):
    scale_factor = 0.25

    # Parse the XML file
    tree = ET.parse(annotations_file)
    root = tree.getroot()

    # list of bounding box coordinates
    newLst = []

    # dictionary mapping image to list of bounding box coordinates
    bounding_dict2 = collections.defaultdict(list)

    # Iterate through each 'image' element in the XML
    for image in root.findall('image'):
        image_id = str(int(image.get('id')) + 1)

        # Iterate through each 'box' element within the 'image'
        for box in image.findall('box'):
            label = box.get('label')
            if label == 'prairie_dog':
                xtl = box.get('xtl')
                ytl = box.get('ytl')
                xbr = box.get('xbr')
                ybr = box.get('ybr')

                bounding_dict2[image_id].append([float(xtl), float(ytl), float(xbr), float(ybr)])
    return bounding_dict2

def count_overlaps_in_dataset(startIdx, device, running_overlap_count, matching_model, coord_dict, method='SPSG'):
    running_total_bounding_boxes = 0

    for i in range(startIdx, 329, 2):
        # read in images
        img1 = cv2.imread(format_image_path(i - 1))
        img2 = cv2.imread(format_image_path(i))

        # convert images to gray-scale
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # downsample images
        scale_factor = 0.25
        down_img1, down_img2 = downsample_img_pair(scale_factor, img1_gray, img2_gray)

        if method == 'SPSG':
            # Convert your images to tensors and move them to the specified device
            image1_tensor = frame2tensor(down_img1, device)
            image2_tensor = frame2tensor(down_img2, device)

            output = matching_model({'image0': image1_tensor, 'image1': image2_tensor})

            # find matching key points
            match_keypoints0, match_keypoints1 = extract_matching_indices(output)

        elif method == 'SIFT':
            if isinstance(SIFT_matching_keypoints(down_img1, down_img2), bool):
                continue
            match_keypoints0, match_keypoints1 = SIFT_matching_keypoints(down_img1, down_img2)

        elif method == 'SPSG and SIFT':
            image1_tensor = frame2tensor(down_img1, device)
            image2_tensor = frame2tensor(down_img2, device)

            output = matching_model({'image0': image1_tensor, 'image1': image2_tensor})

            # find matching key points
            spsg_match_keypoints0, spsg_match_keypoints1 = extract_matching_indices(output)

            if isinstance(SIFT_matching_keypoints(down_img1, down_img2), bool):
                continue
            
            # convert sift keypoints to a set of tuples for fast lookup
            sift_match_keypoints0, sift_match_keypoints1 = SIFT_matching_keypoints(down_img1, down_img2)
            # sift_list1, sift_list2 = np.array(convert_to_list_of_lists(sift_match_keypoints0)), np.array(convert_to_list_of_lists(sift_match_keypoints1))
            # sift_tup_list1, sift_tup_list2 = [tuple(point.ravel()) for point in sift_list1], [tuple(point.ravel()) for point in sift_list2]
            # new_sift_list1, new_sift_list2 = remove_corresponding_duplicates(sift_tup_list1, sift_tup_list2)


            # match_points0, match_points1 = list(spsg_match_keypoints0), list(spsg_match_keypoints1)
            # match_keypoints0, match_keypoints1 = check_membership(set(new_sift_list1), set(new_sift_list2), spsg_match_keypoints0, match_points0, match_points1)

            spsg_list1, spsg_list2 = np.array(convert_to_list_of_lists(spsg_match_keypoints0)), np.array(convert_to_list_of_lists(spsg_match_keypoints1))
            spsg_tup_list1, spsg_tup_list2 = [tuple(point.ravel()) for point in spsg_list1], [tuple(point.ravel()) for point in spsg_list2]
            new_spsg_list1, new_spsg_list2 = remove_corresponding_duplicates(spsg_tup_list1, spsg_tup_list2)
            

            match_points0, match_points1 = list(sift_match_keypoints0), list(sift_match_keypoints1)
            match_keypoints0, match_keypoints1 = check_membership(set(new_spsg_list1), set(new_spsg_list2), spsg_match_keypoints0, match_points0, match_points1)
        

            
        H = filter_with_ransac(match_keypoints0, match_keypoints1)
        # filter matches with RANSAC and return homography
        if isinstance(H, bool):
            continue 
        
        print('Homography matrix: ', H)
        # stitch images together
        dst_padded, src_warped, shifted_transf = warpPerspectivePadded(down_img1, down_img2, H)
        
        # compute padding offsets
        padding_offsets = [generateOffsets(down_img1, H)]
        print(f'Img {i} padding offset: ', padding_offsets)

        raw_total_bounding_boxes = len(coord_dict[f'{i - 1}']) + len(coord_dict[f'{i}'])
        running_total_bounding_boxes += raw_total_bounding_boxes

        # Create a deep copy of bounding_dict2
        bounding_dict2_copy = copy.deepcopy(coord_dict)

        # compute number of overlapping bounding boxes
        overlap_count = overlap_per_img_pair(bounding_dict2_copy[f'{i - 1}'], bounding_dict2_copy[f'{i}'], scale_factor, padding_offsets, shifted_transf)
        running_overlap_count += overlap_count

        # calculate accurate number of boxes and put in dictionary
        print(f'Img {i} overlap count: ', overlap_count)
        print()
        torch.cuda.empty_cache()
    return running_total_bounding_boxes, running_overlap_count

    # clear CUDA memory for next iteration
    #clear_memory()