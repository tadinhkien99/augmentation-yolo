import random
import cv2
import os
import albumentations as A
from convert_format import images_annotations_info
from visualize_image_result import visualize
from yolo_format import images_annotations_yolo_info, write_yolo
from augment_methods import getTransform

if __name__ == "__main__":
    category_id_to_name = {0: "elephant", 1: "frisbee", 2: "kite", 3: "baseball glove"}
    # image_path = "test/test.jpg"
    # label_path = "test/test.txt"
    path = "C:/Users/DELL/Desktop/augment_image_yolo/env/test/"
    all_file = os.listdir(path)
    all_image_path = []
    count = 0

    for file_name in all_file:
        if file_name[-3:] == 'jpg':
            all_image_path.append(path + file_name)
    for file_path in all_image_path:
        image_name = file_path[file_path.rfind('/') + 1:file_path.rfind('.jpg')]
        image_path = file_path
        label_path = file_path.replace('jpg', 'txt')

        image_original = cv2.imread(image_path)
        category_ids, bboxes = images_annotations_info(label_path=label_path, img_height=image_original.shape[0],
                                                       img_width=image_original.shape[1])
        # print(category_ids)
        # print(image_original.shape)
        # transform = A.Compose([
        # A.HorizontalFlip(p=1),
        # A.ShiftScaleRotate(rotate_limit=45, p=1),
        # A.RandomCrop(width=300, height=300, p=1),
        # A.CoarseDropout(p=1),
        # A.GridDropout(p=1),
        # A.Affine(p=1)
        # ],)
        # bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

        # random.seed(7)
        # print(bboxes)
        # transformed = transform(image=image_original, bboxes=bboxes, category_ids=category_ids)
        # img = visualize(
        #     transformed['image'],
        #     transformed['bboxes'],
        #     transformed['category_ids'],
        #     category_id_to_name,
        # )

        # img = visualize(
        #     image_original,
        #     bboxes,
        #     category_ids,
        #     category_id_to_name,
        # )

        # transformed = transform(image=image_original)
        # img = transformed["image"]
        #
        # cv2.imwrite('test_result.jpg', transformed['image'])
        # cv2.imshow("name",img)
        # cv2.waitKey(0)
        #
        category_ids, bboxes = images_annotations_yolo_info(label_path=label_path)
        print(bboxes)
        method_index = 12

        transform_save = getTransform(method_index, img_height=image_original.shape[0],
                                      img_width=image_original.shape[1])
        try:
            transformed_save = transform_save(image=image_original, bboxes=bboxes, category_ids=category_ids)
            print(transformed_save['bboxes'])
            if not os.path.exists("./augmented/" + str(method_index) + "/"):
                os.makedirs("./augmented/" + str(method_index) + "/")
            write_yolo(name="./augmented/" + str(method_index) + "/" + image_name,
                       method_index=str(method_index),
                       coords=transformed_save['bboxes'],
                       category=transformed_save['category_ids'],
                       image=transformed_save['image'])
            print(image_path)
        except:
            count += 1
            print("fail image path is: " + image_path)
    print(count)
