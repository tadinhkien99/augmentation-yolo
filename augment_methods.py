import albumentations as A
import imgaug.augmenters as iaa


def getTransform(loop, img_height, img_width):
    if loop == 0:
        transform = A.Compose([
            A.HorizontalFlip(p=1),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    elif loop == 1:
        transform = A.Compose([
            A.RandomBrightnessContrast(p=1),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    elif loop == 2:
        transform = A.Compose([
            A.MultiplicativeNoise(multiplier=0.5, p=1),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    elif loop == 3:
        transform = A.Compose([
            A.VerticalFlip(p=1)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    elif loop == 4:
        transform = A.Compose([
            A.Blur(blur_limit=(50, 50), p=1)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    elif loop == 5:
        transform = A.Compose([
            A.Transpose(1)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    elif loop == 6:
        transform = A.Compose([
            A.ShiftScaleRotate(rotate_limit=45, p=1)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    elif loop == 7:
        transform = A.Compose([
            A.Perspective (p=1)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    elif loop == 8:
        transform = A.Compose([
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    elif loop == 9:
        transform = A.Compose([
            A.PiecewiseAffine(p=1)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    elif loop == 10:
        transform = A.Compose([
            A.RandomScale(p=1)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

    elif loop == 11:
        transform = A.Compose([
            A.Rotate(limit=45,p=1)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    elif loop == 12:
        transform = A.Compose([
            A.Lambda(p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    # elif loop == 12:
    #     transform = A.Compose([
    #         A.Crop(x_min=0, y_min=0, x_max=100, y_max=100, always_apply=False, p=1.0)
    #         # A.RandomCrop(width=50, height=50,p=1),
    #     ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

    return transform

def getTransform_cut_grid(loop, img_height, img_width):
    if loop == 13:
        transform = iaa.CoarseDropout(0.06, size_percent=0.07)
    elif loop == 14:
        transform = A.Compose([
            A.GridDropout(ratio=0.2,random_offset=True,fill_value=255,holes_number_x=3, holes_number_y=3, p=1),
        ])
    return transform