import cv2
import numpy as np
video_path = "videos/mona-lisa.avi"
cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()

def get_keypoints_descriptors(frame):
    """Находит ключевые точки и их дескрипторы в кадре"""
    detector = cv2.SIFT_create()
    keypoints, descriptors = detector.detectAndCompute(frame, None)
    return keypoints, descriptors

def match_keypoints(desc1, desc2):
    """Находит соответствия между дескрипторами двух кадров"""
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def find_homography(kp1, kp2, matches):
    """Находит гомографию между двумя наборами ключевых точек"""
    if len(matches) < 4:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def draw_tracking_box(frame, H, template_shape):
    """Рисует bounding box на кадре на основе гомографии"""
    if H is None:
        return frame
    h, w = template_shape[:2]
    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    frame_with_box = frame.copy()
    cv2.polylines(frame_with_box, [np.int32(transformed_corners)], True, (0, 255, 0), 2)
    return frame_with_box

while True:
    print("Введите номер желаемого видео:")
    print("1: mona-lisa.avi")
    print("2: mona-lisa-blur.avi")
    print("3: mona-lisa-blur-extra-credit.avi")
    print("4: our-video.avi")
    video_num = int(input())
    if video_num == 1:
        video_path = "videos/mona-lisa.avi"
    elif video_num == 2:
        video_path = "videos/mona-lisa-blur.avi"
    elif video_num == 3:
        video_path = "videos/mona-lisa-blur-extra-credit.avi"
    else:
        video_path = "videos/our-video.mp4"

    cap = cv2.VideoCapture(video_path)
    ret, template_frame = cap.read()
    kp1, desc1 = get_keypoints_descriptors(template_frame)

    while True:
        ret, current_frame = cap.read()
        kp2, desc2 = get_keypoints_descriptors(current_frame)
        if desc1 is not None and desc2 is not None:
            matches = match_keypoints(desc1, desc2)

            if len(matches) > 10:
                H = find_homography(kp1, kp2, matches[:50])
                result_frame = draw_tracking_box(current_frame, H, template_frame.shape)
                matches_img = cv2.drawMatches(template_frame, kp1, result_frame, kp2, matches[:50], None, flags=2)
                cv2.imshow("Соответствия ключевых точек", matches_img)
            else:
                result_frame = current_frame
                cv2.imshow("Соответствия ключевых точек", current_frame)

        cv2.imshow("Трекинг объекта", result_frame if 'result_frame' in locals() else current_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
