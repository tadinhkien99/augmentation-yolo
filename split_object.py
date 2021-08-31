import os
import cv2
from convert_format import images_annotations_info
import shutil

category_id_to_name = {0: "elephant", 1: "frisbee", 2: "kite", 3: "baseball glove"}
SMALL_SIZE = 18*18          # small class <= SMALL_SIZE
MEDIUM_SIZE = 32*32         # SMALL_SIZE < medium class <= MEDIUM_SIZE
LARGE_SIZE = 32*32          # large class > MEDIUM_SIZE

path = "C:/Users/DELL/Desktop/augment_image_yolo/env/test/"
out_path = "C:/Users/DELL/Desktop/augment_image_yolo/env/out_class/"
all_file = os.listdir(path)
all_image_path = []


classes = category_id_to_name.values()
for class_name in classes:
    if not os.path.exists("out_class/small_object/" + class_name +"/"):
        os.makedirs("out_class/small_object/" + class_name +"/")
    if not os.path.exists("out_class/medium_object/" + class_name +"/"):
        os.makedirs("out_class/medium_object/" + class_name +"/")
    if not os.path.exists("out_class/large_object/" + class_name +"/"):
        os.makedirs("out_class/large_object/" + class_name +"/")
    if not os.path.exists("out_class/" + class_name + "_don't_use" + "/"):
        os.makedirs("out_class/" + class_name + "_don't_use" + "/")
    if not os.path.exists("out_class/multiple_class_in_image/"):
        os.makedirs("out_class/multiple_class_in_image/")


image_out_path = "C:/Users/DELL/Desktop/augment_image_yolo/env/out_class/multiple_class_in_image/"
for file_name in all_file:
    if file_name[-3:]=='jpg':
        all_image_path.append(path + file_name)
for file_path in all_image_path:
    image_name = file_path[file_path.rfind('/')+1:file_path.rfind('.jpg')]
    image_path = file_path
    print(image_name)
    label_path = file_path.replace('jpg', 'txt')
    image_original = cv2.imread(image_path)
    category_ids, bboxes = images_annotations_info(label_path=label_path, img_height=image_original.shape[0],
                                                       img_width=image_original.shape[1])
    print(bboxes)
    print(category_ids)
    result = category_ids.count(category_ids[0]) == len(category_ids)
    all_size_in_image = []
    if result:                  #just have one class in a image
        class_name = list(category_id_to_name.values())[category_ids[0]]
        for object in bboxes:
            size_object = object[2]*object[3]
            all_size_in_image.append(size_object)
        # print(all_size_in_image)
        if all(each_size <= SMALL_SIZE for each_size in all_size_in_image):
            img_dst = out_path + "small_object/" + class_name +"/" + image_name + ".jpg"
            label_dst = out_path + "small_object/" + class_name +"/" + image_name + ".txt"
            # print(img_dst)
            shutil.copy(image_path,dst=img_dst)
            shutil.copy(label_path,dst=label_dst)
        elif all((each_size > SMALL_SIZE and each_size <= MEDIUM_SIZE) for each_size in all_size_in_image):
            img_dst = out_path + "medium_object/" + class_name +"/" + image_name + ".jpg"
            label_dst = out_path + "medium_object/" + class_name +"/" + image_name + ".txt"
            # print(img_dst)
            shutil.copy(image_path,dst=img_dst)
            shutil.copy(label_path,dst=label_dst)
        elif all(each_size > MEDIUM_SIZE for each_size in all_size_in_image):
            img_dst = out_path + "large_object/" + class_name +"/" + image_name + ".jpg"
            label_dst = out_path + "large_object/" + class_name +"/" + image_name + ".txt"
            # print(img_dst)
            shutil.copy(image_path,dst=img_dst)
            shutil.copy(label_path,dst=label_dst)
        else:
            img_dst = out_path + class_name + "_don't_use" + "/" + image_name + ".jpg"
            label_dst = out_path + class_name + "_don't_use" + "/" + image_name + ".txt"
            # print(img_dst)
            shutil.copy(image_path,dst=img_dst)
            shutil.copy(label_path,dst=label_dst)
    else:
        shutil.copy(image_path, dst=image_out_path + image_name + ".jpg")
        shutil.copy(label_path, dst=image_out_path + image_name + ".txt")





