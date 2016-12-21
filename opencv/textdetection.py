############################################
#
# Author : Abhisek Mohanty
# Description : This file has methods that contains the logic for generating candidate text in image
#
############################################

import cv2

from CNNs import cnn_recognition
from images.resizeImage import add_padding
from images.resizeImage import maintain_aspect_ratio


def detect(image_path, dimension, cnn_detection, cnn_recog):
    # expects a gray image
    threshold1 = 100
    threshold2 = 200
    img = cv2.imread(image_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, threshold1, threshold2)
    ret, thresh = cv2.threshold(edges, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    texts = []

    for cnt in contours:
        try:
            x, y, w, h = cv2.boundingRect(cnt)
            candidate_text = image[y - 2:y + h + 2, x - 2:x + w + 2]
            texts.append(candidate_text)
            resized_candidate_text = maintain_aspect_ratio(candidate_text, dimension)
            resized_candidate_text = add_padding(resized_candidate_text, dimension)

            # check if it contains a letter, using classifier
            classification = cnn_recognition.classify(resized_candidate_text, cnn_recog)
            print classification
            # if yes, add boundingbox
            if classification[36] != 1:
                parts = cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 1)
        except:
            continue

    cv2.imwrite('files/image.jpg', img)
    print text
    print 'done'
