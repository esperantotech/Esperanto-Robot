import requests
import numpy as np
import torch
import os
from PIL import Image
import supervision as sv
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotate

from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.stack_multiple import Stack
from robosuite.utils.camera_utils import get_real_depth_map,transform_from_pixels_to_world, get_camera_intrinsic_matrix
import cv2

import time 


def get_image(env, model, image_path, depth_image_path=None, image_folder_path='process_images'):
    
    text = "red cube"
    box_threshold = 0.40
    text_threshold = 0.25

    image_source, image = load_image(image_path)
    record_time = 0
    boxes, logits, phrases, record_time = predict(
        model=model,
        image=image,
        caption=text,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        record_time = record_time
    )

    # print (boxes, logits, phrases)
    homography_matrix = get_homography_matrix(image, boxes)
    _, h, w = image.shape
    image_boxes = boxes * np.array([h, w, h, w])
    xyxy = box_convert(boxes=image_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    a, b, c, d = xyxy[0]
    print ('abcd', a, b, c, d)
    world_point = transform_pixel_to_world([(a+c)/2, (b+d)/2], homography_matrix)
    # print (world_point)
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(f'{image_folder_path}/annotated_frame.png', annotated_frame)
    if depth_image_path is not None:
        depth_image_source, depth_image = load_image(depth_image_path)
        annotated_depth_frame = annotate(image_source=depth_image_source, boxes=boxes, logits=logits, phrases=phrases)
        cv2.imwrite(f'{image_folder_path}/annotated_depth_frame.png', annotated_depth_frame)
        _, h_depth, w_depth = depth_image.shape
        depth_image_boxes = boxes * np.array([h_depth, w_depth, h_depth, w_depth])
        xyxy_depth = box_convert(boxes=depth_image_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        a_depth, b_depth, c_depth, d_depth = xyxy_depth[0]
        print ('abcd_depth', a_depth, b_depth, c_depth, d_depth)
    
    ## first get reference depth on table 
    # depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    print (depth_image.shape)
    print (depth_image[0,0,0], depth_image[1,0,0], depth_image[2,0,0])
    print (depth_image[0,25,25], depth_image[1,25,25], depth_image[2,25,25])
    ## take average of first dimension of depth image
    depth_image = depth_image.mean(axis=0)
    print (depth_image.shape)
    reference_depth = depth_image[-1, -1]
    print (reference_depth)
    object_depth = depth_image[int((b_depth+d_depth)/2),int((a_depth+c_depth)/2), ]
    print (torch.abs(object_depth-reference_depth))
    
    
    return world_point

def get_homography_matrix(image, boxes):
    pixel_points = np.array([[209.239, 398.7894535064697], [209.28349494934082, 66.24521017074585], 
                             [114.44417238235474, 208.26282501220703], [399.77078437805176, 66.19623899459839]])
    # world points: cubeA, cubeC, ballA, cubeD
    world_points = np.array([[-0.2, 0.2], [0.15, 0.2], [0, 0.3], [0.15, 0]])
    # calculate homography matrix
    H, _ = cv2.findHomography(pixel_points, world_points)
    return H

def transform_pixel_to_world(pixel_point, H):
    pixel_point_homogeneous = np.append(pixel_point, 1)
    world_point_homogeneous = np.dot(H, pixel_point_homogeneous)
    world_point = world_point_homogeneous / world_point_homogeneous[2]
    return world_point[:2]

    
def process_image(image, cropped_size):

    # crop image into a smaller image for detection
    cropped_w, cropped_h = cropped_size[:2]
    image_h, image_w = image.shape[:2]
    assert cropped_w <= image_w and cropped_h <= image_h, "cropped size should be smaller than image size"
    w_start = int((image_w - cropped_w) / 2)
    h_start = int((image_h - cropped_h) / 2)

    cropped_image = image[h_start:h_start +
                            cropped_h, w_start:w_start+cropped_w]
    
    return cropped_image


def env_image(env,camera_view, image_folder_path):
    obs = env.reset()
    ## save RGB image
    save_image = cv2.flip(obs[f"{camera_view}_image"],1)
    processed_image = Image.fromarray(process_image(save_image, cropped_size=(240,240)))
    processed_image.save(f'{image_folder_path}/processed_image.png')
    
    ## save shown Depth Image
    save_depth_image = cv2.flip(obs[f"{camera_view}_depth"],1)
    save_depth_image = get_real_depth_map(env.sim, save_depth_image)
    processed_depth_image_show = Image.fromarray(process_image((save_depth_image * 255).astype(np.uint8), cropped_size=(240,240)))
    processed_depth_image_show = processed_depth_image_show.convert("L")
    processed_depth_image_show.save(f'{image_folder_path}/processed_depth_image_show.png')
    
    ## save depth information for processing 
    processed_depth_image = Image.fromarray(process_image(save_depth_image, cropped_size=(240,240)))
    processed_depth_image = processed_depth_image.convert("L")
    processed_depth_image.save(f'{image_folder_path}/processed_depth_image.png')
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config_path = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    weight_path = 'gd_weights/groundingdino_swint_ogc.pth'
    model = load_model(config_path, weight_path, device)
    image_folder_path = 'process_images'
    camera_view = "birdview"
    
    env = Stack(
        robots="Kinova3",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=camera_view,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=20,
        camera_depths = True,
    )
    
    env_image(env, camera_view, image_folder_path)
    image_path = f'{image_folder_path}/processed_image.png'
    depth_image_path = f'{image_folder_path}/processed_depth_image.png'
    get_image(env, model, image_path, depth_image_path, image_folder_path)
    env.close()

if __name__ == "__main__":
    ## This file still needs to be fixed to get depth information from build-in stereo camera
    main()

