import cv2
import mediapipe as mp
import numpy as np
from multiprocessing import Process

def mediapipe_baseline(video_path):
    # 初始化 Mediapipe 的 Face Mesh 模型
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 获取面部关键点的索引，额头的大致区域可以通过这些索引确定
    # 例如，Face Mesh 中第10号点是额头的中心点
    FOREHEAD_POINTS = [10, 338, 297, 332, 284, 251, 389, 356]
    cap = cv2.VideoCapture(video_path)

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

                    # 可选：绘制额头区域的特征点 red
                    for (x, y) in forehead_landmarks:
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                        
                    # 可选：如果你想可视化每个面部特征点 green
                    for idx, (x, y) in enumerate(landmarks):
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            cv2.imshow('MediaPipe', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def haar_baseline(video_path):
    
    # 加载人脸检测的 Haar 特征分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)

    # Shi-Tomasi 角点检测的参数
    feature_params = dict(maxCorners=500, 
                          qualityLevel=0.3, 
                          minDistance=7, 
                          blockSize=7)

    # lucas kanade 光流法参数
    lk_params = dict(winSize=(15, 15), 
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 随机颜色用于绘制跟踪点
    color = np.random.randint(0, 255, (100, 3))

    # 初始化变量
    old_gray = None
    p0 = None  # 初始特征点
    face_box = None  # 人脸的框

    # 额外增加的高度比例，用于包含额头
    forehead_factor = 1  # 可以根据需要调整这个比例

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 如果还没有初始点，检测人脸并提取角点
        if p0 is None:
            # 检测人脸
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                # 增加人脸框的高度以包含额头
                h = int(h * forehead_factor)
                y = max(0, y - int(h * (forehead_factor - 1)))  # 调整 y 位置，确保不超出上边界

                # 保存人脸的初始边界框
                face_box = (x, y, w, h)
                # 在人脸区域内寻找特征点
                face_roi = gray[y:y+h, x:x+w]
                p0 = cv2.goodFeaturesToTrack(face_roi, mask=None, **feature_params)
                if p0 is not None:
                    # 将特征点的坐标从人脸区域转换为全局坐标
                    p0[:, :, 0] += x
                    p0[:, :, 1] += y
                    old_gray = gray.copy()

        # 如果有特征点，使用 KLT 算法跟踪特征点
        if p0 is not None:
            # 计算光流
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

            # 选择成功跟踪的点
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # 在帧上绘制跟踪的特征点
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    # 转换坐标为整数
                    a, b = int(a), int(b)
                    c, d = int(c), int(d)
                    # 绘制跟踪点
                    frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                    # 绘制运动轨迹
                    frame = cv2.line(frame, (a, b), (c, d), color[i].tolist(), 2)

                # 更新旧的特征点和帧
                old_gray = gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

                # 更新人脸边界框坐标
                x_min = np.min(good_new[:, 0])  # 获取所有点的最小 x 坐标
                y_min = np.min(good_new[:, 1])  # 获取所有点的最小 y 坐标
                x_max = np.max(good_new[:, 0])  # 获取所有点的最大 x 坐标
                y_max = np.max(good_new[:, 1])  # 获取所有点的最大 y 坐标

                # 额外增加高度以包含额头部分
                y_min = max(0, y_min - int((y_max - y_min) * (forehead_factor - 1)))  # 顶部扩展
                face_box = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
            else:
                # 如果跟踪失败，重新检测人脸
                p0 = None

        # 如果有更新的人脸框，用矩形框将人脸框出
        if face_box is not None:
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 显示结果帧
        cv2.imshow('Haar Baseline', frame)

        # 按 'q' 键退出
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    


def haar_klt(video_path):
    import cv2
    import numpy as np

    # 加载人脸检测的 Haar 特征分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)

    # Shi-Tomasi 角点检测的参数
    feature_params = dict(maxCorners=500, 
                          qualityLevel=0.3, 
                          minDistance=7, 
                          blockSize=7)

    # lucas kanade 光流法参数
    lk_params = dict(winSize=(15, 15), 
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 随机颜色用于绘制跟踪点
    color = np.random.randint(0, 255, (100, 3))

    # 初始化变量
    old_gray = None
    p0 = None  # 初始特征点
    face_box = None  # 人脸的框

    # 额外增加的高度比例，用于包含额头
    forehead_factor = 1  # 可以根据需要调整这个比例

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 如果还没有初始点，检测人脸并提取角点
        if p0 is None:
            # 检测人脸
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                # 增加人脸框的高度以包含额头
                h = int(h * forehead_factor)
                y = max(0, y - int(h * (forehead_factor - 1)))  # 调整 y 位置，确保不超出上边界

                # 保存人脸的初始边界框
                face_box = (x, y, w, h)
                # 在人脸区域内寻找特征点
                face_roi = gray[y:y+h, x:x+w]
                p0 = cv2.goodFeaturesToTrack(face_roi, mask=None, **feature_params)
                if p0 is not None:
                    # 将特征点的坐标从人脸区域转换为全局坐标
                    p0[:, :, 0] += x
                    p0[:, :, 1] += y
                    old_gray = gray.copy()

        # 如果有特征点，使用 KLT 算法跟踪特征点
        if p0 is not None:
            # 计算光流
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

            # 选择成功跟踪的点
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # 在帧上绘制跟踪的特征点
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    # 转换坐标为整数
                    a, b = int(a), int(b)
                    c, d = int(c), int(d)
                    # 绘制跟踪点
                    frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                    # 绘制运动轨迹
                    frame = cv2.line(frame, (a, b), (c, d), color[i].tolist(), 2)

                # 更新旧的特征点和帧
                old_gray = gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

                # 更新人脸边界框坐标
                x_min = np.min(good_new[:, 0])  # 获取所有点的最小 x 坐标
                y_min = np.min(good_new[:, 1])  # 获取所有点的最小 y 坐标
                x_max = np.max(good_new[:, 0])  # 获取所有点的最大 x 坐标
                y_max = np.max(good_new[:, 1])  # 获取所有点的最大 y 坐标

                # 额外增加高度以包含额头部分
                y_min = max(0, y_min - int((y_max - y_min) * (forehead_factor - 1)))  # 顶部扩展
                face_box = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
            else:
                # 如果跟踪失败，重新检测人脸
                p0 = None

        # 如果有更新的人脸框，用矩形框将人脸框出
        if face_box is not None:
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
        cv2.imshow('harr and KLT', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # path = './DATASET_1/5-gt/vid.avi'
    path = './DATASET_2/subject/vid.avi'
    
    haar = Process(target=haar_baseline, args=(path,))
    mediapipe = Process(target=mediapipe_baseline, args=(path,))
    klt = Process(target=haar_klt, args=(path,))
    
    haar.start()
    mediapipe.start()
    klt.start()
    
    haar.join()
    mediapipe.join()
    klt.join()