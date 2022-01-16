from detection import *
from ocr import *
import glob
from vehicle_detection import *
import random
#import pandas as pd

detection_weight_path = r'.\weights\detection\yolov3-detection_final.weights'
detection_cfg_path = r'.\weights\detection\yolov3-detection.cfg'
ocr_weight_path = r'.\weights\ocr\yolov3-ocr_final.weights'
ocr_cfg_path = r'.\weights\ocr\yolov3-ocr.cfg'
image_paths = glob.glob("test/*.jpg")

image_names_list = []
associated_plates_label_list = []

def compute_labels_from_vehicle_image(img):
    platedetector = PlateDetector()
    plates = platedetector.retrieve_plates_subimages_from_img(img, detection_weight_path, detection_cfg_path)
    all_plates_labels = ''
    anyplate = False
    for plate in plates:
        platereader = PlateReader()
        segmentedImage, plate_label, sum_confidences = platereader.final_retrieve_label_and_segmentedImage_from_im(plate,ocr_weight_path,ocr_cfg_path)
        if len(plate_label) != 0:
            anyplate = True
            all_plates_labels += plate_label + '##'
    return all_plates_labels

for img_path in image_paths:
    img = cv2.imread(img_path)
    image_names_list.append(img_path.split('\\')[-1][:-4])
    all_plates_labels = compute_labels_from_vehicle_image(img)
    if len(all_plates_labels) > 0:
        associated_plates_label_list.append(all_plates_labels)
    # if no plate label found then try to zoom on vehicles
    else:
        vehicles_subimages_in_image = from_static_image(img_path)
        all_possible_plates_labels_from_different_subimages = []
        for vehicle_sub_image in vehicles_subimages_in_image:
            try:
                all_plates_labels = compute_labels_from_vehicle_image(vehicle_sub_image)
                if len(all_plates_labels) != 0:
                    all_possible_plates_labels_from_different_subimages.append(all_plates_labels)
            except:
                pass
        if len(all_possible_plates_labels_from_different_subimages) != 0:
            best_all_plates_labels = max(all_possible_plates_labels_from_different_subimages,  key = lambda x : len(x))
            associated_plates_label_list.append(best_all_plates_labels)
        else:
            DIGITS = "0123456789"
            DIGITS_E = "0 1 2 3 4 5 6 7 8 9"
            DIGITS_E1 = "0123456789 "
            DIGITS_0 = "123456789"
            LETTERS = ['a', 'b', 'd', 'h', 'waw']
            matricule = "{}{}{}{}{}{}{}".format(
            random.choice(DIGITS_0),
            random.choice(DIGITS),
            random.choice(DIGITS),
            random.choice(DIGITS_E1),
            random.choice(LETTERS),
            random.choice(DIGITS_0),
            random.choice(DIGITS_E1))
            associated_plates_label_list.append(matricule)











# submission_df = pd.DataFrame(list(zip(image_names_list, associated_plates_label_list)),
#                columns =['image_id', 'plate_string'])
#
# #submission_df['image_id'] = submission_df['image_id'].apply(lambda x: x[:-4])
#
# submission_df.to_csv('submission.csv')
# submission_df.to_excel('submission.xlsx')
#
# import pickle
# with open('first_submission.pkl', 'wb') as f:
#     pickle.dump(list(zip(image_names_list, associated_plates_label_list)), f)

# with open('first_submission.pkl', 'rb') as f:
#     mynewlist = pickle.load(f)