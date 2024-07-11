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
import cv2


def get_image(model, image_path):
    
    text = "green bottle"
    box_threshold = 0.35
    text_threshold = 0.25

    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    
    print (boxes, logits, phrases)
    homography_matrix = get_homography_matrix(image, boxes)
    h, w = image.shape[:2]
    image_boxes = boxes * np.array([w, h, w, h])
    xyxy = box_convert(boxes=image_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    a, b, c, d = xyxy[0]
    # print (a, b, c, d)
    world_point = transform_pixel_to_world([(a+c)/2, (b+d)/2], homography_matrix)
    print (world_point)
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("annotated_frame.png", annotated_frame)

def get_homography_matrix(image, boxes):
    # h, w = image.shape[:2]
    # boxes = boxes * np.array([w, h, w, h])
    # xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    # print (xyxy)
    # a, b, c, d = xyxy[0]
    # print (a, b, c, d)
    # print ((a+c)/2, (b+d)/2)
    # input()
    # pixel points: red cube, blue cube, red ball, yellow cube
    pixel_points = np.array([[209.285, 1.4955], [209.29, 0.248], [114.523, 0.7812], [399.76, 0.24815]])
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
    sample_num = 1

    for i in range(sample_num):

        # crop image into a smaller image for detection
        cropped_w, cropped_h = cropped_size[:2]
        image_h, image_w = image.shape[:2]
        assert cropped_w <= image_w and cropped_h <= image_h, "cropped size should be smaller than image size"
        w_start = int((image_w - cropped_w) / 2)
        h_start = int((image_h - cropped_h) / 2)
        if i == 0:
            w_start = int((image_w - cropped_w) / 2)
            h_start = int((image_h - cropped_h) / 2)
        else:
            w_start = np.random.randint(0, image_w - cropped_w)
            h_start = np.random.randint(0, image_h - cropped_h)

        cropped_image = image[h_start:h_start +
                                cropped_h, w_start:w_start+cropped_w]
    
    return cropped_image
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config_path = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    weight_path = 'gd_weights/groundingdino_swint_ogc.pth'
    model = load_model(config_path, weight_path, device)
    
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
    )
    
    obs = env.reset()
    save_image = cv2.flip(obs[f"{camera_view}_image"],1)
    env.close()
    processed_image = Image.fromarray(process_image(save_image, cropped_size=(240,240)))
    processed_image.save('processed_image.png')
    image_path = 'processed_image.png'
    get_image(model, image_path)

if __name__ == "__main__":
    main()

