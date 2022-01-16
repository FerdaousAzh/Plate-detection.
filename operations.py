import numpy as np
import cv2
import glob


"""
l'invertion pour detecter les images de string "maroc" car elles ont un fond noir
pour detecter egalement les images de lettre jim
mais si deja on a pu predire au moins 6 lettres alors on fait rien
si on trouve j sans inversion alors on garde puis on inverse pr trouver les chiffres
"""
def inverte(img):
    img = (255-img)
    #cv2.imshow(name, img)
    #cv2.imwrite(name, img)
    return img

#image_paths = ['m\\m1.jpg','m\\m2.jpg','m\\m3.jpg','m\\m4.jpg', 'm\\523_0_plate.jpg']
# for img_path in image_paths:
#     img = cv2.imread(img_path)
#     inverte(img, img_path[:-4]+'inverted.jpg')

def get_contour_areas(contours):
    all_areas = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)

    return all_areas

def largest_contour(img_path):
    image = cv2.imread(img_path)
    largest_contour_img(image)

def largest_contour_img(image):
    original_image = image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 200)

    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #cv2.destroyAllWindows()

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    largest_item = sorted_contours[0]

    rect = cv2.boundingRect(largest_item)
    x,y,w,h = rect
    #box = cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
    cropped = image[y: y+h, x: x+w]
    #cv2.imshow("cropped", cropped)
    return cropped

    #cv2.drawContours(original_image, largest_item, -1, (255, 0, 0), 10)
    # cv2.waitKey(0)
    #cv2.imshow('Largest Object', original_image)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #cv2.imshow('Largest Object', largest_item)


# img_path = 'm\\m4inverted.jpg'
# largest_item = largest_contour(img_path)

def controller(img, brightness=255,
               contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:

        if brightness > 0:
            shadow = brightness

            max = 255

        else:

            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)

    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)



    return cal

def test_many_contrast_brightness_configurations(img):
    image = img.copy()
    contrasted_images = []
    brightness_range = 500
    contrast_range = 250
    brightness_percentage_min = 20
    step_brightness_percentage = 10
    contrast_percentage_max = 100
    step_contrast_percentage = 10
    for brightness_percentage in range(brightness_percentage_min, 100, step_brightness_percentage):
        brightness = brightness_percentage*brightness_range/100
        relative_contrast_min_percentage = brightness_percentage
        relative_contrast_percentage_max = max(relative_contrast_min_percentage + 2*step_contrast_percentage, contrast_percentage_max)
        for relative_contrast_percentage in range(relative_contrast_min_percentage, relative_contrast_percentage_max+step_contrast_percentage , step_contrast_percentage):
            contrast = relative_contrast_percentage*contrast_range/100
            contrasted_images.append(controller(image, brightness, contrast))
    return contrasted_images


# # brightness = 170
# # contrast = 200
# original = cv2.imread(r'.\need contrast\452_0_plate.jpg')
# # contrasted_image = controller(original, brightness,contrast)
# # cv2.imwrite(r'.\need contrast\467_0_plate_contrasted.jpg',contrasted_image)
# contrasted_images = test_many_contrast_brightness_configurations(original)
# for i,x in enumerate(contrasted_images):
#     cv2.imwrite(r'.\need contrast\contrasted\contrasted_'+str(i)+'.jpg', x)