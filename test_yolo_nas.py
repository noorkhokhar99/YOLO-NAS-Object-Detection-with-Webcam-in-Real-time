import super_gradients

yolo_nas = super_gradients.training.models.get("yolo_nas_l", pretrained_weights="coco").cuda()
yolo_nas.predict("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg").show()