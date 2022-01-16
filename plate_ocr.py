from ocr import *
import glob
import matplotlib.pyplot as plt

weight_path = r'.\weights\ocr\yolov3-ocr_final.weights'
cfg_path = r'.\weights\ocr\yolov3-ocr.cfg'

def show_image(image):
    fig, ax1 = plt.subplots(1)
    ax1.imshow(image)
    plt.show()

image_paths = glob.glob("plates1/*.jpg")
image_paths = glob.glob("m/*.jpg")
image_paths = ['need contrast\\new\\534_0_plate.jpg']
id = 0
for img_path in image_paths:
    platereader = PlateReader()
    img, _, _, _ = platereader.load_image(img_path)
    #segmentedImage, plate_label = platereader.retrieve_label_and_segmentedImage_from(img_path, weight_path, cfg_path)
    segmentedImage, plate_label, sum_confidences = platereader.final_retrieve_label_and_segmentedImage_from_im(img, weight_path, cfg_path)
    #cv2.imshow(plate_label, segmentedImage)
    #show_image(segmentedImage)
    #cv2.imwrite(img_path, segmentedImage)
    cv2.imwrite(img_path[:-4]+str(id)+"_plate_"+plate_label+".jpg", segmentedImage)
    id += 1
