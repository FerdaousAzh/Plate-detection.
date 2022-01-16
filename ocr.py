import cv2
import pytesseract
import numpy as np
import glob
from detect_red import *
from operations import *

class PlateReader:
    def load_model(self, weight_path: str, cfg_path: str):
        self.net = cv2.dnn.readNet(weight_path, cfg_path)
        with open("classes-ocr.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layers_names = self.net.getLayerNames()
        self.output_layers = [self.layers_names[i[0]-1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        return img, height, width, channels

    def read_plate(self, img):
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        return blob, outputs
    
    def get_boxes(self, outputs, width, height, threshold=0.3):
        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > threshold:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
          
        return boxes, confidences, class_ids
    
    def draw_labels(self, boxes, confidences, class_ids, img):
        segmented_image = img.copy()
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
        font = cv2.FONT_HERSHEY_PLAIN
        c = 0
        characters = []
        sum_confidences = 0
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[i % len(self.colors)]
                cv2.rectangle(segmented_image, (x,y), (x+w, y+h), color, 3)
                confidence = round(confidences[i], 3) * 100
                sum_confidences += confidence
                cv2.putText(segmented_image, str(confidence) + "%", (x, y - 6), font, 1, color, 2)
                characters.append((label, x))
        characters.sort(key=lambda x:x[1])
        plate = ""
        for l in characters:
            plate += l[0]

        return segmented_image, plate, sum_confidences

    def final_retrieve_label_and_segmentedImage_from_im(self, img, weight_path, cfg_path):
        """
        detecter le label
        si ya un caractère rouge
        o	si label de taille >= 6 ne rien faire
        o	sinon
            	si jim figure dans le label alors lettre finale = jim, sinon = m
            	invertir
            	largest contour pour enlever « duster » sauf pour jim
            	predire les chiffres puis rajouter lettre finale
        """
        segmentedImage, label, sum_confidences = self.retrieve_label_and_segmentedImage_from_im(img, weight_path, cfg_path)
        if len(label) >= 6:
            return segmentedImage, label, sum_confidences
        else:
            is_morocco_character = is_plate_with_morocco_string_img(img)
            if not is_morocco_character:
                best_found = segmentedImage, label, sum_confidences
                images_with_different_contrast_n_brightness_from_img = test_many_contrast_brightness_configurations(img)
                for image_contrasted in images_with_different_contrast_n_brightness_from_img:
                    new_segmentedImage, new_label, new_sum_confidences = self.retrieve_label_and_segmentedImage_from_im(image_contrasted,
                                                                                                            weight_path,
                                                                                                            cfg_path)
                    if new_sum_confidences > sum_confidences:
                        best_found = new_segmentedImage, new_label, new_sum_confidences
                return best_found
            else:
                img = inverte(img)
                if 'j' in label:
                    predicted_letter = 'j'
                else:
                    predicted_letter = 'm'
                    img = largest_contour_img(img)
                segmentedImage, label, sum_confidences = self.retrieve_label_and_segmentedImage_from_im(img, weight_path, cfg_path)
                if predicted_letter not in label:
                    label = label+predicted_letter
                return segmentedImage, label, sum_confidences

    def retrieve_label_and_segmentedImage_from_im(self, img, weight_path, cfg_path):
        self.load_model(weight_path, cfg_path)
        height, width, channels = img.shape
        blob, outputs = self.read_plate(img)
        boxes, confidences, class_ids = self.get_boxes(outputs, width, height, threshold=0.02)
        segmentedImage, label, sum_confidences = self.draw_labels(boxes, confidences, class_ids, img)
        return segmentedImage, label, sum_confidences

    def retrieve_label_and_segmentedImage_from(self, img_path, weight_path, cfg_path):
        self.load_model(weight_path, cfg_path)
        img, height, width, channels = self.load_image(img_path)
        blob, outputs = self.read_plate(img)
        boxes, confidences, class_ids = self.get_boxes(outputs, width, height, threshold=0.02)
        segmentedImage, label, sum_confidences = self.draw_labels(boxes, confidences, class_ids, img)
        return segmentedImage, label, sum_confidences

    def arabic_chars(self, index):
        if (index == ord('a')):
            return "أ".encode("utf-8")
        
        if (index == ord('b')):
            return "ب".encode("utf-8")

        if (index == 2 * ord('w') + ord('a') or index == ord('w')):
            return "و".encode("utf-8")

        if (index == ord('d')):
            return "د".encode("utf-8")
            
        if (index == ord('h')):
            return "ه".encode("utf-8")

        if (index == ord('c') + ord('h')):
            return "ش".encode("utf-8")

    def tesseract_ocr(self, image, lang="eng", psm=7):
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-l {} --psm {} -c tessedit_char_whitelist={}".format(lang, psm, alphanumeric)
        return pytesseract.image_to_string(image, config=options)
