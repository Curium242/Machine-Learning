#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ultralytics import YOLO
import cv2
import random
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# In[2]:


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


# In[3]:


model = YOLO("yolo-Weights/yolov8n.pt")


# In[4]:


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


# In[5]:


available_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
selected_font = random.choice(available_fonts)
print("Selected Font:", selected_font)


# In[6]:


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Generate a random color for each box
            box_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            org = (x1, y1 + 20)  # Adjust text position
            font_path = selected_font
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            fontScale = 1.2
            color = (255, 255, 255)  # White text color
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Unique Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break


# In[ ]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:




