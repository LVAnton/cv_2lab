import cv2
import numpy as np

sift = cv2.SIFT_create()
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

def get_features(frame):
    """Находим ключевые точки и дескрипторы в кадре"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def find_good_matches(desc1, desc2):
    """Находим хорошие соответствия между дескрипторами"""
    if desc1 is None or desc2 is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def find_homography_from_matches(kp1, kp2, matches):
    """Находим гомографию по совпавшим точкам"""
    if len(matches) < 4:
        return None
    src_points = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_points = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H

def get_corners(frame_shape):
    """Получаем углы рамки (по размеру кадра)"""
    h, w = frame_shape[:2]
    return np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

def init_tracking_points(gray_frame, box_corners):
    """Инициализируем точки для отслеживания оптическим потоком"""
    mask = np.zeros_like(gray_frame, dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(box_corners)], 255)
    points = cv2.goodFeaturesToTrack(gray_frame, mask=mask, maxCorners=150,
                                     qualityLevel=0.01, minDistance=5, blockSize=7)
    return points

def draw_matches_image(template_frame, kp1, current_frame, kp2, matches, box=None):
    """Рисуем совпадения точек между кадрами"""
    img_matches = cv2.drawMatches(template_frame, kp1, current_frame, kp2,
                                  matches[:50], None, flags=2)
    if box is not None:
        w = template_frame.shape[1]
        offset = np.array([[w, 0]], dtype=np.float32)
        box_right = box + offset
        cv2.polylines(img_matches, [np.int32(box_right)], True, (0, 255, 0), 2)

    return img_matches

while True:
    print("Выберите видео для отслеживания:")
    print("1. mona-lisa.avi")
    print("2. mona-lisa-blur.avi")
    print("3. mona-lisa-blur-extra-credit.avi")
    print("4. our-video.mp4")

    choice = input("Ваш выбор (1-4): ").strip()

    if choice == "1":
        video_path = "videos/mona-lisa.avi"
    elif choice == "2":
        video_path = "videos/mona-lisa-blur.avi"
    elif choice == "3":
        video_path = "videos/mona-lisa-blur-extra-credit.avi"
    else:
        video_path = "videos/our-video.mp4"

    cap = cv2.VideoCapture(video_path)
    ret, template_frame = cap.read()
    kp_template, desc_template = get_features(template_frame)
    template_gray = cv2.cvtColor(template_frame, cv2.COLOR_BGR2GRAY)
    corners = get_corners(template_frame.shape)
    tracking_box = corners.copy()
    tracking_points = init_tracking_points(template_gray, tracking_box)
    prev_gray = template_gray

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result_frame = frame.copy()
        if tracking_points is not None and len(tracking_points) > 0:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, tracking_points, None, **lk_params
            )

            if new_points is not None:
                good_new = new_points[status == 1]
                good_old = tracking_points[status == 1]
                if len(good_new) >= 4:
                    H, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 3.0)
                    if H is not None:
                        tracking_box = cv2.perspectiveTransform(tracking_box, H)
                        tracking_points = good_new.reshape(-1, 1, 2)

        if tracking_points is None or len(tracking_points) < 10:
            kp_current, desc_current = get_features(frame)
            if desc_current is not None:
                matches = find_good_matches(desc_template, desc_current)
                if len(matches) >= 10:
                    H = find_homography_from_matches(kp_template, kp_current, matches)
                    if H is not None:
                        tracking_box = cv2.perspectiveTransform(corners, H)
                        tracking_points = init_tracking_points(gray, tracking_box)

        if tracking_box is not None:
            cv2.polylines(result_frame, [np.int32(tracking_box)], True, (0, 255, 0), 2)

            kp_current, desc_current = get_features(frame)
            if desc_current is not None:
                matches = find_good_matches(desc_template, desc_current)
                matches_img = draw_matches_image(
                    template_frame, kp_template, result_frame,
                    kp_current, matches, tracking_box
                )
                cv2.imshow("Совпадения точек", matches_img)

        prev_gray = gray.copy()
        cv2.imshow("Улучшенное отслеживание объекта", result_frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('e'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
