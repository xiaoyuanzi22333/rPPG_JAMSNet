from UBFC2 import UBFC2_data
import cv2
import numpy as np
import pandas as pd
import os
import mediapipe as mp
from tqdm import tqdm


src_pth = './DATASET_2'
tgt_pth = './DATASET_2_transform'

# 初始化 Mediapipe 的 Face Mesh 模型
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def mediapipe_baseline(data: UBFC2_data, subject_tgt_pth, generate=True):

    
    # 面部关键点的索引，Face Mesh 中第10号点是额头的中心点
    FOREHEAD_POINTS = [10, 338, 297, 332, 284, 251, 389, 356]
    video_path = data.getVideo()
    cap = cv2.VideoCapture(video_path)

    video_crop = []
    # 设置 Face Mesh 模型参数
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # 最多检测1张脸
        refine_landmarks=True,  # 提供更精细的面部特征点，例如眼睛、嘴唇的内外轮廓
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # 如果检测到面部
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 获取所有的面部特征点坐标
                    img_h, img_w, _ = frame.shape
                    landmarks = np.array(
                        [[int(point.x * img_w), int(point.y * img_h)] for point in face_landmarks.landmark])

                    # 找到面部特征点的最小和最大坐标，计算边界框
                    x_min = np.min(landmarks[:, 0])
                    y_min = np.min(landmarks[:, 1])
                    x_max = np.max(landmarks[:, 0])
                    y_max = np.max(landmarks[:, 1])

                    # 使用额头特征点扩展边界框
                    forehead_landmarks = landmarks[FOREHEAD_POINTS]
                    forehead_y_min = np.min(forehead_landmarks[:, 1])  # 额头的最小 y 坐标

                    # 调整边界框，使其包含额头的上部
                    y_min = min(y_min, forehead_y_min - int((y_max - y_min) * 0.1))  # 向上扩展边界框
                    img = cv2.resize(frame[y_min:y_max,x_min:x_max,:], (128,192))
                    video_crop.append(img)
                    # print(len(video_crop))
                    
    # generate 150width and 30fps small video if generate=True
    cap.release()
    for i in range(int(len(video_crop)/150)):
        crop_video_folder_pth = os.path.join(subject_tgt_pth, 'video_'+str(i))
        if not os.path.exists(crop_video_folder_pth):
            os.mkdir(crop_video_folder_pth)
        
        video_generate = video_crop[i:i+150]
        crop_video_pth = os.path.join(crop_video_folder_pth,'vid.avi')
        
        if generate:
            size = (128,192)
            videoWriter = cv2.VideoWriter(
                crop_video_pth,
                cv2.VideoWriter_fourcc(*'XVID'),  # 编码器
                30,
                size
            )
            for img in video_generate:
                videoWriter.write(img)
            videoWriter.release()
            gtTrace, gtHR = data.getGT()
            np.save(crop_video_folder_pth+'/gtTrace.npy',gtTrace[i:i+150])
            np.save(crop_video_folder_pth+'/gtHR', gtHR[i:i+150])
        
        


def transfrom():
    for file_name in tqdm(os.listdir(src_pth)):
        
        # create subject folder for each dataset
        subject_tgt_path = os.path.join(tgt_pth, file_name)
        subject_src_path = os.path.join(src_pth, file_name)
        if not os.path.exists(subject_tgt_path):
            os.mkdir(subject_tgt_path)
        
        subject = UBFC2_data(subject_src_path)
        video_crop = mediapipe_baseline(subject, subject_tgt_path)

        



if __name__ == "__main__":
        
    transfrom()
    print("transformed")