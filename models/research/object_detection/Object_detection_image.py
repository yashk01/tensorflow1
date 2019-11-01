######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'test1.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 6

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value

# vidcap = cv2.VideoCapture("parking.mp4")
vidcap = cv2.VideoCapture("http://172.16.215.97:8080/video/mjpeg")
fps = int(vidcap.get(cv2.CAP_PROP_FPS))
count = 0
while vidcap.isOpened():
    success, image = vidcap.read()
    if success and (count%(2*fps)==0):
        # cv2.destroyAllWindows()
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(image,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8,min_score_thresh=0.60)

        # cv2.imshow("l", image)

        # cv2.line(image, (151, 62), (257, 131), (0,255,0), 6)
        # cv2.line(image, (151, 62), (20, 250), (0,255,0), 6)
        # cv2.line(image, (20, 250), (118, 328), (0,255,0), 6)
        # cv2.line(image, (257, 131), (118, 328), (0,255,0), 6)

        # 151, 62
        # 118, 328


        # cv2.line(image, (232, 197), (276, 220), (0,255,0), 6)
        # cv2.line(image, (232, 197), (165, 294), (0,255,0), 6)
        # cv2.line(image, (203, 328), (165, 294), (0,255,0), 6)
        # cv2.line(image, (203, 328), (276, 220), (0,255,0), 6)

        # 232, 197
        # 203,328

        for i in range(len(scores[0])):
            if scores[0][i] > 0.60 :


            # finding where the cetner lies
            
                width = image.shape[1]
                height = image.shape[0]

                ymin = boxes[0][i][0]*height
                xmin = boxes[0][i][1]*width
                ymax = boxes[0][i][2]*height
                xmax = boxes[0][i][3]*width
                # print ('Top left')
                # print (xmin,ymin,)
                # print ('Bottom right')
                # print (xmax,ymax)

                rect1center = int((xmin+xmax)/2)
                rect2center = int((ymax+ymin)/2)

                # print(rect1center, rect2center)

                cv2.line(image, (int(rect1center),int(rect2center)), (0,0), (0, 255, 0), 3)

        # if (rect1center in range(118, 151)) and (rect2center in range(62, 328)):
        #     print("full")


        # cv2.imwrite("C:/Users/Yash/Desktop/%d.png" % count, image)
    # count+=1




# # image = cv2.imread(PATH_TO_IMAGE)
# image_expanded = np.expand_dims(image, axis=0)

# # Perform the actual detection by running the model with the image as input
# (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_expanded})

# # Draw the results of the detection (aka 'visulaize the results')

# vis_util.visualize_boxes_and_labels_on_image_array(image,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8,min_score_thresh=0.60)



# for i in range(len(scores[0])):
#     if scores[0][i] > 0.60 :


#     # finding where the cetner lies

#         width = image.shape[1]
#         height = image.shape[0]

#         ymin = boxes[0][i][0]*height
#         xmin = boxes[0][i][1]*width
#         ymax = boxes[0][i][2]*height
#         xmax = boxes[0][i][3]*width
#         print ('Top left')
#         print (xmin,ymin,)
#         print ('Bottom right')
#         print (xmax,ymax)

#         rect1center = ((xmin+xmax)/2)
#         rect2center = ((ymax+ymin)/2)

#         print(rect1center, rect2center)

#         cv2.line(image, (int(rect1center),int(rect2center)), (0,0), (0, 255, 0), 3)

# All the results have been drawn on image. Now display the image.
# cv2.imshow('Object detector', image)

# Press any key to close the image
cv2.waitKey(0)
vidcap.release()
# Clean up
cv2.destroyAllWindows()