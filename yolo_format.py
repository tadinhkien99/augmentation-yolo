import cv2


def images_annotations_yolo_info(label_path):
    annotations = []
    category_ids = []
    all_lines = []
    images = []
    image_id = 0
    annotation_id = 1   # In COCO dataset format, you must start annotation id with '1'


    with open(label_path,"r", encoding="utf-8") as label_file:
        all_lines = label_file.readlines()
        label_file.close()

    for line1 in all_lines:
        annotation = []
        label_line = line1.split()
        # print(label_line)
        category_ids.append(int(label_line[0]))
        x_center = float(label_line[1])
        y_center = float(label_line[2])
        width = float(label_line[3])
        height = float(label_line[4])
        annotation.append(x_center)
        annotation.append(y_center)
        annotation.append(width)
        annotation.append(height)
        annotations.append(annotation)
    return category_ids, annotations

def write_yolo(name, method_index, coords, category, image):
    cv2.imwrite(name+ '_' + method_index +'.jpg', image)
    with open(name+ '_' + method_index +'.txt', "w", encoding="utf-8") as f:
        for i in range(len(category)):
            # for x in coords[i]:
            f.write("%s %s %s %s %s \n" % (category[i], coords[i][0], coords[i][1],
                                           coords[i][2], coords[i][3]))
        f.close()