import numpy as np
import cv2
import glob

"""
detecter sil sagit dune plaque de string "maroc" car elle a des lettres en rouges
"""

def is_plate_with_morocco_string(img_path):
    img = cv2.imread(img_path)
    return is_plate_with_morocco_string_img(img)

def is_plate_with_morocco_string_img(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower mask(0 - 10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    # upper mask(170 - 180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    # join my masks
    mask = mask0 + mask1
    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0
    # or your HSV image, which I * believe * is what you want
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0
    #cv2.imshow('mine2', output_hsv)

    #limage doit etre localisée et non nulle apres mask
    #non nullité
    is_null = True
    red_pixel_percentage = len(output_hsv[np.where(output_hsv>10)])/img.size
    if red_pixel_percentage > 0.006:
        is_null = False
    #localisation
    range_x = np.where(output_hsv>10)[1]
    is_localized = False
    if len(range_x) !=0:
        range_x_dist = max(range_x)-min(range_x)
        percentage_width_of_red_part = 100*range_x_dist/img.shape[1]
        if percentage_width_of_red_part > 42:
            is_localized = False
        else:
            is_localized = True
    if (not is_null) and is_localized:
        return True
    else:
        return False

# maroc_plates = []
# image_paths = glob.glob("plates1/*.jpg")
# for img_path in image_paths:
#     #print(img_path)
#     #print(is_plate_with_morocco_string(img_path))
#     if is_plate_with_morocco_string(img_path):
#         maroc_plates.append(img_path)

# img_path = 'plates1\\569_0_plate.jpg'
# img_path = 'm\\m4.jpg'
# print(is_plate_with_morocco_string(img_path))
