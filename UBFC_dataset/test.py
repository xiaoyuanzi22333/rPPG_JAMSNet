import cv2
import numpy as np


v1 = './DATASET_2_transform/subject8/video_7/vid.avi'
v2 = './DATASET_2_transform/subject8/video_0/vid.avi'

gttrace = './DATASET_2_transform/subject8/video_0/gtHR.npy'


# cap1 = cv2.VideoCapture(v1)
# cap2 = cv2.VideoCapture(v2)

# succ1,frame1 = cap1.read()
# succ2,frame2 = cap2.read()

# # print(frame1)

# cv2.imwrite('./frame1.jpg', frame1)
# cv2.imwrite('./frame2.jpg', frame2)

trace = np.load(gttrace)
print(trace)