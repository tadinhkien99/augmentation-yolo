import cv2
import os
from convert_format import images_annotations_info
from augment_methods import getTransform, getTransform_cut_grid


category_id_to_name = {0: "elephant", 1: "frisbee", 2: "kite", 3: "baseball glove"}
# image_path = "test/test.jpg"
# label_path = "test/test.txt"
path = "C:/Users/DELL/Desktop/augment_image_yolo/env/test/"
all_file = os.listdir(path)
all_image_path = []
count=0

for file_name in all_file:
    if file_name[-3:]=='jpg':
        all_image_path.append(path + file_name)
for file_path in all_image_path:
    image_name = file_path[file_path.rfind('/')+1:file_path.rfind('.jpg')]
    image_path = file_path
    label_path = file_path.replace('jpg', 'txt')

    image_original = cv2.imread(image_path)
    category_ids, bboxes = images_annotations_info(label_path=label_path, img_height=image_original.shape[0],
                                                    img_width=image_original.shape[1])

    method_index = 13
    transform_save = getTransform_cut_grid(method_index, img_height=image_original.shape[0],
                                           img_width=image_original.shape[1])
    try:
        # transformed_image = transform_save(image=image_original)['image'] #if use 14
        transformed_image = transform_save(image=image_original)       #if use 13

        if not os.path.exists("./augmented/" + str(method_index) + "/"):
            os.makedirs("./augmented/" + str(method_index) + "/")
        cv2.imwrite("./augmented/" + str(method_index) + "/"+image_name + '_' + str(method_index) + '.jpg', transformed_image)

        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            f.close()
            # lines = [l for l in lines if "ROW" in l]
        f1 = open("./augmented/" + str(method_index) + "/"+image_name + '_' + str(method_index) + '.txt', "w", encoding='utf-8')
        f1.writelines(lines)
        f1.close()
        print(image_path)
    except:
        count += 1
        print("fail image path is: " + image_path)
print(count)









