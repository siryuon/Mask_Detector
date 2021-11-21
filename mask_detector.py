from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import imutils
import time
import argparse
import numpy as np
import cv2
import os

def mask_detector(frame, network, model):

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    network.setInput(blob)
    detections = network.forward()
    
    faces = []
    locations = []
    predicts = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > minimum_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locations.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        predicts = model.predict(faces, batch_size=32)
    
    return (locations, predicts)

#DNN Module
face_detector = "./face_detector/"
prototxt = face_detector + "deploy.prototxt"
weights = face_detector + "res10_300x300_ssd_iter_140000.caffemodel"
network = cv2.dnn.readNet(prototxt, weights)

mask_detector_model = "mymodel.h5"
model = load_model(mask_detector_model)

vs = cv2.VideoCapture(0)

minimum_confidence = 0.5

while True:
    ret, frame = vs.read()
    
    frame = imutils.resize(frame, width=500)
    
    (locations, predicts) = mask_detector(frame, network, model)

    for (box, predict) in zip(locations, predicts):
        (startX, startY, endX, endY) = box

        (mask, without_mask) = predict
        
        label = "Mask" if mask > without_mask else "No Mask"
        
        if label == "Mask" and max(mask, without_mask) * 100 >= 70:
            color = (0, 255, 0)
        elif label == "No Mask" and max(mask, without_mask) * 100 >= 70:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
 
    cv2.imshow("Mask Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("x"):
        break

vs.release()
cv2.destroyAllWindows()
