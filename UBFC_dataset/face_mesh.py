import cv2
import mediapipe as mp
import numpy as np

# 初始化 Mediapipe 的 Face Mesh 模型
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 定义面部关键点的索引
FOREHEAD_POINT = 10   # 额头中心
CHIN_POINT = 152      # 下巴
LEFT_EAR_POINT = 234  # 左耳
RIGHT_EAR_POINT = 454 # 右耳

video_path = "./DATASET_1/5-gt/vid.avi"
cap = cv2.VideoCapture(video_path)

# 设置 Face Mesh 模型参数
with mp_face_mesh.FaceMesh(
    max_num_faces=1,  # 最多检测1张脸
    refine_landmarks=True,  # 提供更精细的面部特征点
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            continue

        # 翻转图像，转换为 RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 处理图像并返回结果
        results = face_mesh.process(rgb_frame)

        # 如果检测到面部
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 获取所有的面部特征点坐标
                img_h, img_w, _ = frame.shape
                landmarks = np.array(
                    [[int(point.x * img_w), int(point.y * img_h)] for point in face_landmarks.landmark])

                # 提取额头、下巴、左耳、右耳的点
                forehead = landmarks[FOREHEAD_POINT]  # 额头中心点
                chin = landmarks[CHIN_POINT]          # 下巴点
                left_ear = landmarks[LEFT_EAR_POINT]  # 左耳点
                right_ear = landmarks[RIGHT_EAR_POINT]# 右耳点

                # 计算边界框的最小和最大坐标
                x_min = min(forehead[0], chin[0], left_ear[0], right_ear[0])
                y_min = min(forehead[1], left_ear[1], right_ear[1])
                x_max = max(forehead[0], chin[0], left_ear[0], right_ear[0])
                y_max = max(chin[1], left_ear[1], right_ear[1])

                # 绘制边界框
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                # 可选：绘制关键特征点
                cv2.circle(frame, tuple(forehead), 5, (0, 255, 0), -1)  # 额头
                cv2.circle(frame, tuple(chin), 5, (0, 255, 255), -1)    # 下巴
                cv2.circle(frame, tuple(left_ear), 5, (255, 0, 0), -1)  # 左耳
                cv2.circle(frame, tuple(right_ear), 5, (0, 0, 255), -1) # 右耳

        # 显示结果
        cv2.imshow('MediaPipe Face Mesh with Forehead', frame)

        # 按 'q' 键退出
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()