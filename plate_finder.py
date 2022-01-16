from detection import *
weight_path = r'.\weights\detection\yolov3-detection_final.weights'
cfg_path = r'.\weights\detection\yolov3-detection.cfg'


image_paths = glob.glob("test/*.jpg")
#image_paths = [r'.\need contrast\468_1_plate_contrasted.jpg']
for img_path in image_paths:
    platedetector = PlateDetector()
    plates = platedetector.retrieve_plates_subimages_from(img_path, weight_path, cfg_path)
    if len(plates) != 0:
        j=0
        for plate in plates:
            #cv2.imwrite('plates1\\'+img_path[5:-4]+'_'+str(j)+'_plate.jpg', plate)
            #cv2.imwrite(img_path[:-4]+'_'+str(j)+'_plate.jpg', plate)
            cv2.imwrite(img_path[:-4] + '_' + str(j) + '_plate.jpg', plate)
            j += 1
    else:
        print('couldnt find plate in '+img_path)
