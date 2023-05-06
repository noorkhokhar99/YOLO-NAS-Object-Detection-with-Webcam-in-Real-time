import cv2
import torch
from super_gradients.training  import models
from super_gradients.common.object_names import Models

#models = models.get(Models.YOLO_NAS_l,pretrained_w)
model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")

model = model.to("cuda" if torch.cuda.is_available() else "cpu")

model.predict_webcam()
ouputs.show()
