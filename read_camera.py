#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:wchao118
@license: Apache Licence 
@file: read_camera.py 
@time: 2019/06/26
@contact: wchao118@gmail.com
@software: PyCharm 
"""

import cv2
import os
from main import api
from PIL import Image

save_path = '/home/ai/Desktop/project3/datasets/wang'
font = cv2.FONT_HERSHEY_SIMPLEX
detector = cv2.CascadeClassifier('/home/ai/.local/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
print('ok')
id = 0
while True:
    id += 1
    ret, img = cap.read()
    if id % 10 == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_face = img[y:y+w, x:x+h]
            name = api(crop_face)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, name, (x, y), font, 1.2, (255, 0, 0), 2)
        cv2.imshow('test', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
