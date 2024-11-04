import cv2
import numpy as np

file = np.load('./DATASET_2/train/subject1/video_0/rep_0.npy')
file_1 = np.load('./DATASET_2/train/subject1/video_0/rep_1.npy')
file_2 = np.load('./DATASET_2/train/subject3/video_0/rep_2.npy')
file_gt = np.load('./DATASET_2/train/subject1/video_0/gtTrace.npy')

file = np.array(file)
file_1 = np.array(file_1)
file_2 = np.array(file_2)
file_gt = np.array(file_gt)
file_gt.resize((150,1))

print(file.shape)
print(file_1.shape)
print(file_2.shape)
print(file_gt.shape)

