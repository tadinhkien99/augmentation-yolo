import cv2
import argparse
import json
import numpy as np


def images_annotations_info(label_path, img_height, img_width):
    annotations = []
    category_ids = []
    all_lines = []
    images = []
    image_id = 0
    annotation_id = 1   # In COCO dataset format, you must start annotation id with '1'


    with open(label_path,"r", encoding="utf-8") as label_file:
        all_lines = label_file.readlines()
        label_file.close()


    # yolo format - (class_id, x_center, y_center, width, height)
    # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)

    for line1 in all_lines:
        annotation = []
        label_line = line1.split()
        # print(label_line)
        category_ids.append(int(label_line[0]))
        x_center = float(label_line[1])
        y_center = float(label_line[2])
        width = float(label_line[3])
        height = float(label_line[4])


        int_x_center = int(img_width*x_center)
        int_y_center = int(img_height*y_center)
        int_width = int(img_width*width)
        int_height = int(img_height*height)

        min_x = abs(int_x_center-int_width/2)
        min_y = abs(int_y_center-int_height/2)
        width = int_width
        height = int_height
        annotation.append(min_x)
        annotation.append(min_y)
        annotation.append(width)
        annotation.append(height)
        annotations.append(annotation)

    return category_ids, annotations