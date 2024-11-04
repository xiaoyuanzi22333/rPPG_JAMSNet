import cv2
import mediapipe as mp
import numpy as np


def mediapipe_baseline():
    # 初始化 Mediapipe 的 Face Mesh 模型
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 获取面部关键点的索引，额头的大致区域可以通过这些索引确定
    # 例如，Face Mesh 中第10号点是额头的中心点
    FOREHEAD_POINTS = [10, 338, 297, 332, 284, 251, 389, 356]
    path = './example.jpg'

    # 设置 Face Mesh 模型参数
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # 最多检测1张脸
        refine_landmarks=True,  # 提供更精细的面部特征点，例如眼睛、嘴唇的内外轮廓
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        frame = cv2.imread(path)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # 如果检测到面部
        if results.multi_face_landmarks:
            print("detected")
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

                # 没有额头强化
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                

                # 使用额头特征点扩展边界框
                forehead_landmarks = landmarks[FOREHEAD_POINTS]
                forehead_y_min = np.min(forehead_landmarks[:, 1])  # 额头的最小 y 坐标

                # 调整边界框，使其包含额头的上部
                y_min = min(y_min, forehead_y_min - int((y_max - y_min) * 0.1))  # 向上扩展边界框

                # 绘制扩展的边界框
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                crop = frame[y_min:y_max,x_min:x_max,:]
                cv2.imwrite('crop.jpg', crop)
                print(crop.shape)

                # 可选：绘制额头区域的特征点 red
                for (x, y) in forehead_landmarks:
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                    
                # 可选：如果你想可视化每个面部特征点 green
                for idx, (x, y) in enumerate(landmarks):
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        cv2.imwrite('MediaPipe.jpg', frame)
        
        
        
if __name__ == "__main__":
    mediapipe_baseline()